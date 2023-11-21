import copy
import json
import os
import random
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import util.misc as utils
from data.multiple_dataset import CombinedDataset
from tokenizers.utils import batched_space_sampler, np_to_tensor
from transformer_network import TransformerNetwork
from transformer_network_test_set_up import state_space_list


def load_config_from_json(json_path):
    with open(json_path, "r") as f:
        config = json.load(f)
    return config


def set_seed(seed=3407):
    """
    set random seed to reproduce results
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(self, args):
        set_seed()
        self.args = args
        self.train_dataset = CombinedDataset(time_sequence_length=self.args["time_sequence_length"])
        self.args = utils.init_distributed_mode(self.args)
        self.checkpoint_dir, self.tensorboard_dir = self.make_log_dir(self.args["log_dir"])
        if self.args["distributed"]:
            self.sampler_train = DistributedSampler(self.train_dataset, shuffle=True)

        self.args["checkpoint_dir"] = self.checkpoint_dir
        self.writer_train = SummaryWriter(self.tensorboard_dir, flush_secs=5)
        self._action_space = spaces.Dict(
            OrderedDict(
                [
                    (
                        "first_three",
                        spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
                    ),
                    (
                        "middle_three",
                        spaces.Box(low=-np.pi, high=np.pi, shape=(3,), dtype=np.float32),
                    ),
                    (
                        "final_one",
                        # spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                        spaces.Discrete(2)
                    ),
                ]
            )
        )
        self.args["action_space"] = str(self._action_space)
        with open(
            os.path.join(self.checkpoint_dir, self.train_name + ".json"), "w"
        ) as json_file:
            json.dump(self.args, json_file)
        json_file.close()
        self.device = torch.device(self.args["device"])
        self.train_step = 0

    def train(self):
        print("training")

        # Set random seed for reproducibility
        set_seed()

        # Create dataloader based on distributed or single-machine settings
        if self.args["distributed"]:
            # Batch sampler for distributed training
            batch_sampler_train = torch.utils.data.BatchSampler(
                self.sampler_train, self.args["batch_size"], drop_last=True
            )
            train_dataloader = DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler_train,
                num_workers=self.args["batch_size"],
            )
        else:
            # DataLoader for single-machine training
            train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.args["batch_size"],
                num_workers=0,
                shuffle=True,
                drop_last=True,
            )

        # Initialize the TransformerNetwork based on specified configurations
        network_configs = self.args["network_configs"]
        # Modify network configuration based on specific settings
        network_configs["time_sequence_length"] = self.args["time_sequence_length"]
        # network_configs["num_encoders"] = len(self.args["cam_view"])
        network_configs["token_embedding_size"] = network_configs["token_embedding_size_per_image"]
        del network_configs["token_embedding_size_per_image"]
        # network_configs["using_proprioception"] = self.args["using_proprioception"]
        network_configs["input_tensor_space"] = state_space_list()[0]
        network_configs["output_tensor_space"] = self._action_space
        network = TransformerNetwork(**network_configs)
        network.to(self.device)
        network_without_ddp = network

        # Load model weights, optimizer, scheduler settings, resume from checkpoints if specified
        if self.args["resume"]:
            checkpoint = torch.load(
                self.args["resume_from_checkpoint"], map_location="cpu"
            )
        total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
        print("number of model params:", total_params)
        total_size_bytes = total_params * 4
        # Parameter is in torch.float32，Each parameter takes 4 bytes
        total_size_mb = round(total_size_bytes / (1024 * 1024), 2)
        print("model size: ", total_size_mb, " MB")

        # Configuration based on distributed or single-machine setup
        if self.args["distributed"]:
            # DistributedDataParallel setup
            network = torch.nn.parallel.DistributedDataParallel(
                network, device_ids=[self.args["gpu"]], find_unused_parameters=False
            )
            network_without_ddp = network.module
            optimizer = torch.optim.AdamW(
                network_without_ddp.parameters(), lr=self.args["lr"]
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer, **self.args["scheduler_configs"]
            )
            if self.args["resume"]:
                network_without_ddp.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            # Single-machine setup
            optimizer = torch.optim.AdamW(network.parameters(), lr=self.args["lr"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer, **self.args["scheduler_configs"]
            )
            if self.args["resume"]:
                network.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        epoch_start = checkpoint["epoch"] if self.args["resume"] else 0
        for e in range(epoch_start, self.args["epochs"]):
            network.train()
            with tqdm(
                train_dataloader, dynamic_ncols=True, desc="train"
            ) as tqdmDataLoader:
                # for _, (obs, action) in enumerate(tqdmDataLoader):
                for _, item in enumerate(tqdmDataLoader):
                    # Perform training steps
                    obs = item['observation']
                    action = item['action']
                    optimizer.zero_grad()
                    network_without_ddp.set_actions(
                        self.dict_to_device(action, self.device)
                    )
                    network_state = batched_space_sampler(
                        network_without_ddp._state_space,
                        batch_size=self.args["batch_size"],
                    )
                    network_state = np_to_tensor(network_state)
                    # if self.args["using_proprioception"]:
                    #     obs = self.calc_fk(obs)
                    output_actions, network_state = network(
                        self.dict_to_device(obs, self.device),
                        self.dict_to_device(network_state, self.device),
                    )

                    loss = network_without_ddp.get_actor_loss().mean()

                    loss.backward()
                    optimizer.step()

                    # Logging metrics during training
                    if utils.is_main_process():
                        # Log loss, epoch, and learning rate
                        self.writer_train.add_scalar(
                            tag="loss_ce",
                            global_step=self.train_step,
                            scalar_value=loss.cpu().data.numpy(),
                            walltime=time.time(),
                        )
                        self.writer_train.add_scalar(
                            tag="epoch",
                            global_step=self.train_step,
                            scalar_value=e,
                            walltime=time.time(),
                        )
                        self.writer_train.add_scalar(
                            tag="lr",
                            global_step=self.train_step,
                            scalar_value=optimizer.state_dict()["param_groups"][0][
                                "lr"
                            ],
                            walltime=time.time(),
                        )
                    self.train_step += 1
                    tqdmDataLoader.set_postfix(
                        ordered_dict={
                            "epoch": e,
                            "train_name": self.train_name[-5:],
                            "gpu_memory_used": str(
                                round(
                                    torch.cuda.max_memory_allocated() / (1024**3), 2
                                )
                            )
                            + " GB",
                            "loss": loss.item(),
                            "lr": optimizer.state_dict()["param_groups"][0]["lr"],
                        }
                    )

            # Perform validation at specified intervals
            if (e + 1) % self.args["val_interval"] == 0:
                checkpoint_filename = os.path.join(
                    self.checkpoint_dir, str(e) + "-checkpoint.pth"
                )
                checkpoint = {
                    "model_state_dict": network_without_ddp.state_dict()
                    if self.args["distributed"]
                    else network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": e,
                }
                utils.save_on_master(checkpoint, checkpoint_filename)
                if self.args["distributed"]:
                    # Barrier synchronization for distributed training
                    print(
                        f"Process {torch.distributed.get_rank()} has reached the end of epoch {e}."
                    )
                    torch.distributed.barrier()
                    # self.val(
                    #     network_without_ddp=network_without_ddp,
                    #     epoch=e,
                    #     val_dataset=self.val_dataset,
                    #     sampler_val=self.sampler_val,
                    # )
                    print(
                        f"Process {torch.distributed.get_rank()} has reached the end of val."
                    )
                    torch.distributed.barrier()
                # else:
                #     self.val(
                #         network_without_ddp=network,
                #         epoch=e,
                #         val_dataset=self.val_dataset,
                #     )
            scheduler.step()

    # @torch.no_grad()
    # def val(self, network_without_ddp, epoch, val_dataset, sampler_val=None):
    #     # Create directories to store validation results if they don't exist
    #     if not os.path.isdir(os.path.join(self.checkpoint_dir, "val_results")) and utils.is_main_process():
    #         os.mkdir(os.path.join(self.checkpoint_dir, "val_results"))
    #
    #     # Set up dataloader based on distributed or single-machine settings
    #     if self.args["distributed"]:
    #         val_dataloader = DataLoader(
    #             val_dataset, batch_size=1, sampler=sampler_val, drop_last=False
    #         )
    #     else:
    #         val_dataloader = DataLoader(
    #             val_dataset, batch_size=1, shuffle=False, drop_last=False
    #         )
    #
    #     network_without_ddp.eval()
    #
    #     # Perform validation without gradient calculation
    #     val_loss_func = nn.CrossEntropyLoss(reduction="mean")
    #     val_loss_func_mae = nn.L1Loss(reduction="mean")
    #     val_losses = []
    #     gt_one_episode = []
    #     model_output_one_episode = []
    #     # Loop through the validation dataset
    #     for idx, (obs, action) in tqdm(
    #         enumerate(val_dataloader), desc="validation", total=len(val_dataset) // self.args['world_size']
    #     ):
    #         # Initialize network state
    #         network_state = batched_space_sampler(
    #             network_without_ddp._state_space, batch_size=1
    #         )
    #         network_state = np_to_tensor(network_state)
    #
    #         # Reset network state
    #         for k, v in network_state.items():
    #             network_state[k] = torch.zeros_like(v)
    #
    #         action_predictions_logits = []
    #         output_actions = []
    #
    #         # Infer actions for each timestep
    #         # if self.args["using_proprioception"]:
    #         #     obs = self.calc_fk(obs)
    #         for i_ts in range(self.args["time_sequence_length"]):
    #             ob = self.retrieve_single_timestep(obs, i_ts)
    #             output_action, network_state = network_without_ddp(
    #                 self.dict_to_device(ob, self.device),
    #                 self.dict_to_device(network_state, self.device),
    #             )
    #             output_actions.append(output_action)
    #             action_predictions_logits.append(
    #                 network_without_ddp._aux_info["action_predictions_logits"]
    #             )
    #
    #         # Get ground truth actions
    #         gt_actions = network_without_ddp._action_tokenizer.tokenize(action)
    #
    #         # Process predictions and ground truth actions
    #
    #         # since when calculating cross entrophy, the class probability needs to be at the second dimension
    #         # we move the class probability from the last dimension to second dimension
    #         action_predictions_logits = (
    #             torch.cat(action_predictions_logits, dim=0)
    #             .unsqueeze(0)
    #             .permute(0, 3, 1, 2)
    #         )
    #         gt_one_episode.append(gt_actions)
    #         model_output_one_episode.append(action_predictions_logits.argmax(1))
    #
    #         # Handle end of episode scenario
    #         if gt_actions[0, -1, 0] == 2:
    #             # gt_actions[0, -1, 0] is the terminate signal for current episode, 2 indicates the end of episode
    #             # whtn terminate signal is triggered, we write this episode's test results into files
    #             gt_one_episode = torch.cat(gt_one_episode).cpu().data.numpy()
    #             model_output_one_episode = (
    #                 torch.cat(model_output_one_episode).cpu().data.numpy()
    #             )
    #
    #             # Visualize and store episode results
    #             if utils.is_main_process():
    #                 if not os.path.isdir(os.path.join(self.checkpoint_dir, "pics")):
    #                     os.mkdir(os.path.join(self.checkpoint_dir, "pics"))
    #                 if not os.path.isdir(
    #                     os.path.join(self.checkpoint_dir, "val_results")
    #                 ):
    #                     os.mkdir(os.path.join(self.checkpoint_dir, "val_results"))
    #             fn = (
    #                 "epoch_"
    #                 + str(epoch)
    #                 + "_step_"
    #                 + str(idx)
    #                 + "_gpu_"
    #                 + str(self.args["gpu"])
    #                 + ".pdf"
    #             )
    #             fn = os.path.join(self.checkpoint_dir, "val_results", fn)
    #             self.visualize(gt_one_episode, model_output_one_episode, fn)
    #             print("result written into: ", fn)
    #             gt_one_episode = []
    #             model_output_one_episode = []
    #
    #         # Calculate validation loss metrics
    #         val_loss = (
    #             val_loss_func(action_predictions_logits, gt_actions.to(self.device))
    #             .cpu()
    #             .data.numpy()
    #         )
    #         val_loss_mae = val_loss_func_mae(
    #             action_predictions_logits.argmax(1).float(), gt_actions.to(self.device)
    #         ).cpu()
    #         val_losses.append(val_loss)
    #
    #         # Log validation metrics
    #         if utils.is_main_process():
    #             self.writer_val.add_scalar(
    #                 tag="loss_ce",
    #                 global_step=self.val_step,
    #                 scalar_value=val_loss,
    #                 walltime=time.time(),
    #             )
    #             self.writer_val.add_scalar(
    #                 tag="loss_mae",
    #                 global_step=self.val_step,
    #                 scalar_value=val_loss_mae.data.numpy(),
    #                 walltime=time.time(),
    #             )
    #             self.writer_val.add_scalar(
    #                 tag="epoch",
    #                 global_step=self.val_step,
    #                 scalar_value=epoch,
    #                 walltime=time.time(),
    #             )
    #         self.val_step += 1
    #
    #     # Close the writer and return validation losses
    #     self.writer_val.close()
    #     return val_losses

    def multi_test_in_sim_env(self, epoch, network, optimizer, scheduler):
        pass

    def evaluate(self):
        pass

    def calculate_completion_rate(self, file_path):
        pass

    @torch.no_grad()
    def visualize(self, all_gt, all_output, fn):
        all_output = all_output[:, -1, :]
        all_gt = all_gt[:, -1, :]
        title = [
            "terminate_episode_l1_error: ",
            "cmd_pos_x_l1_error: ",
            "cmd_pos_y_l1_error: ",
            "cmd_pos_z_l1_error: ",
            "cmd_rot_x_l1_error: ",
            "cmd_rot_y_l1_error: ",
            "cmd_rot_z_l1_error: ",
            "cmd_gripper_l1_error: ",
        ]
        plt.figure(figsize=(22, 12))
        for i in range(8):
            c = utils.generate_random_color()
            plt.subplot(2, 4, i + 1)
            val_loss = F.l1_loss(
                torch.from_numpy(all_output[:, i]).float(),
                torch.from_numpy(all_gt[:, i]).float(),
            )
            plt.title(title[i] + str(val_loss.cpu().data.numpy()))
            plt.plot(all_gt[:, i], c=c, label="gt")
            plt.plot(all_output[:, i], c=c, linestyle="dashed", label="output")
            plt.xlabel("timesteps")
            plt.ylabel("action_tokens")
            plt.grid()
            plt.legend()
        plt.savefig(fn, format="pdf")
        plt.clf()
        plt.close()

    def retrieve_single_timestep(self, dict_obj, idx):
        """
        get all the values in the [dict_obj] at index [idx]
        v[:, idx], all the values in the dictionary at second dimension needs to be same
        """
        dict_obj_return = copy.deepcopy(dict_obj)
        for k, v in dict_obj.items():
            dict_obj_return[k] = v[:, idx]
        return dict_obj_return

    def dict_to_device(self, dict_obj, device):
        """
        put all the values in the [dict_obj] to [device]
        """
        for k, v in dict_obj.items():
            assert isinstance(v, torch.Tensor)
            dict_obj[k] = v.to(device)
        return dict_obj

    def make_log_dir(self, log_dir):
        """
        making the log directory
        the file structure of log dir should be:
            [log_dir]
                [log_0]
                [log_1]
                ...
                [tensorboard_logs]
                    [log_0]
                    [log_1]
                    ...
        Parameters:
        - log_dir(str): root directory storing all the logs
        Returns:
        - checkpoint_dir(str): log directory for this sepcific training
        - checkpoint_dir(str): tensorboard_dir directory for this sepcific training
        """

        id = str(time.time()).split(".")[0]
        train_name = id
        self.train_name = train_name
        if not os.path.isdir(os.path.join(log_dir)):
            os.makedirs(os.path.join(log_dir))
        checkpoint_dir = os.path.join(log_dir, train_name)
        if not os.path.isdir(os.path.join(log_dir, "tensorboard_logs")):
            os.mkdir(os.path.join(log_dir, "tensorboard_logs"))
        tensorboard_dir = os.path.join(log_dir, "tensorboard_logs", train_name)
        if utils.is_main_process():
            os.mkdir(checkpoint_dir)
        return checkpoint_dir, tensorboard_dir


if __name__ == "__main__":
    args = load_config_from_json("config.json")
    trainer = Trainer(args)
    if args["mode"] == "train":
        trainer.train()
    else:
        trainer.evaluate()