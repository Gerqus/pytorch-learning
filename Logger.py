import datetime
from pathlib import Path
import sys
from termcolor import colored
import torch
import logging
import torch.nn as nn
from sizeformatter import sizeof_fmt

class Logger:
    def __init__(self, model_name: str):
        model_name = model_name.replace(" ", "_").upper()
        self.logger = logging.getLogger(model_name)
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        Path(f"logs/{model_name}").mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=f"logs/{model_name}/{date_time}.log",
            encoding="utf-8",
            filemode="a",
            level=logging.INFO,
            format="%(message)s",
        )
        handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(handler)

    def log_initial_setup(
        self,
        model: nn.Module,
        epoch_length,
        expected_end_loss,
        stale_loss_treshold_change,
        do_nudge,
        patience_window,
        data_precision_type,
        learning_rate,
        scheduler_gamma,
        nudge_magnitude,
    ):
        data_precision_bytes = data_precision_type.itemsize * 8
        model_device = model.parameters().__next__().device
        params_count = sum(p.numel() for p in model.parameters())
        memory_allocated = (
            torch.cuda.memory_allocated(model_device)
            if model_device.type == "cuda"
            else params_count * data_precision_bytes
        )
        memory_allocated_formatted = sizeof_fmt(memory_allocated)
        self.logger.info(
            f"""Starting training:
- precision\t{data_precision_bytes} bytes
- device\t{torch.cuda.get_device_name(model_device) if model_device.type == 'cuda' else model_device}

- target loss\t{expected_end_loss}
- stale loss treshold change\t{stale_loss_treshold_change}

- epoch length\t{epoch_length}
- initial learning rate\t{learning_rate}
- scheduler gamma\t{scheduler_gamma}

- nudging\t{'ON' if do_nudge else 'OFF'}
- nudge magnitude\t{nudge_magnitude}
- patience window\t{patience_window}

- model parameters count\t{params_count}
- model size in memory\t{memory_allocated_formatted}
- model overview\n{model}
"""
        )

    def log_model_training_results(self, raw_output, processed_output, expected_output):
        correct = (processed_output == expected_output).float().sum()
        self.logger.info(f"Correct predictions count: {correct}/{len(expected_output)}")
        expected_values_list = torch.squeeze(expected_output, 1).tolist()
        self.logger.info(f"Y:\n{expected_values_list}")
        raw_output_list = ["%.4f" % i for i in torch.squeeze(raw_output, 1).tolist()]
        self.logger.info(f"Raw values:\n{raw_output_list}")
        processed_output_list = torch.squeeze(processed_output, 1).tolist()
        colored_output_list = [
            colored(i, "green") if i == j else colored(i, "red")
            for i, j in zip(processed_output_list, expected_values_list)
        ]
        self.logger.info(f"Processed values: {processed_output_list}")
        print(f"[{', '.join(colored_output_list)}]")
