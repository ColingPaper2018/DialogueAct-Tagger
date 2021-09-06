from taggers.transformer_tagger import TransformerTagger, BERT
from config import TransformerConfig
from .trainer import Trainer
from utils import stringify_tags
from corpora.corpus import Utterance
from pathlib import Path
import os

from typing import List
import torch
from torchtext.data import BucketIterator
import logging
import random
from typing import Tuple

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ISO_DA")


class TransformerTrainer(Trainer):
    """
    Dialogue Act Trainer using the Hugging Face transformer architecture
    """

    def __init__(self, config: TransformerConfig, corpora_list: List[Tuple[type, str]]):
        Trainer.__init__(self, config, config.taxonomy, corpora_list)
        for c in corpora_list:
            try:
                self.corpora.append(c[0](c[1], config.taxonomy))
            except Exception as e:
                logger.warning(f"Corpus {c[0]} not loaded. {e}")

    def save_checkpoint(self, model, model_name, valid_loss):
        """
        Support method to save a checkpoint
        :param valid_loss: validation loss from last epoch
        :param model: model to save
        :return:
        """
        state_dict = {"model_state_dict": model.state_dict(), "valid_loss": valid_loss}

        path = Path(os.path.dirname(self.config.out_folder))
        path.mkdir(parents=True, exist_ok=True)

        torch.save(state_dict, f"{self.config.out_folder}/{model_name}.pt")
        logger.info(f"Model saved to ==> {self.config.out_folder}")

    def train_transformer(
        self,
        train_set: List[Utterance],
        valid_set: List[Utterance],
        n_classes: int,
        model_name: str,
    ) -> BERT:
        """
        Trains a transformer for a classification task. Returns an untrained model
         if no data is provided or only one label is present
        :param train_set: dataset to use for the training
        :param valid_set: dataset to use for the validation
        :param n_classes: number of classes to classify
        :param model_name: name of the model in the models dictionary
        :return: the trained model
        """
        model = BERT(n_classes).to(self.config.device)
        optimizer = self.config.optimizer(model.parameters(), lr=self.config.lr)

        if len(train_set) == 0 or n_classes < 2:
            logger.info(
                f"Skipping training for {model_name}; dataset length was {len(train_set)}, "
                f"n_classes={n_classes}"
            )
            self.save_checkpoint(model, model_name, 1.0)
            return model

        train_iter = BucketIterator(
            TransformerTagger.build_features(train_set, self.config),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.Text),
            device=self.config.device,
            train=True,
            sort=True,
            sort_within_batch=True,
        )

        valid_iter = BucketIterator(
            TransformerTagger.build_features(valid_set, self.config),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.Text),
            device=self.config.device,
            train=True,
            sort=True,
            sort_within_batch=True,
        )
        eval_every = max(len(train_iter) // 2, 1)
        best_valid_loss = float("Inf")

        # initialize running values
        running_loss = 0.0
        valid_running_loss = 0.0
        global_step = 0
        train_loss_list = []
        valid_loss_list = []
        global_steps_list = []

        logger.info(f"Training model {model_name}")

        # training loop
        model.train()
        for epoch in range(self.config.n_epochs):
            for (text, label), _ in train_iter:
                print("text")
                labels = label.type(torch.LongTensor)
                labels = labels.to(self.config.device)
                output = model(text, labels)
                # print("calculating loss")
                # loss, _ = output
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                #
                # # update running values
                # running_loss += loss.item()
                # global_step += 1
                # print("evaluation")
                # # evaluation step
                # if global_step % eval_every == 0:
                #     model.eval()
                #     with torch.no_grad():
                #
                #         # validation loop
                #         for (vtext, vlabels), _ in valid_iter:
                #             vlabels = vlabels.type(torch.LongTensor)
                #             vlabels = vlabels.to(self.config.device)
                #             vtext = vtext.type(torch.LongTensor)
                #             vtext = vtext.to(self.config.device)
                #             output = model(vtext, vlabels)
                #             loss, _ = output
                #
                #             valid_running_loss += loss.item()
                #
                #     # evaluation
                #     average_train_loss = running_loss / eval_every
                #     average_valid_loss = valid_running_loss / len(valid_iter)
                #     train_loss_list.append(average_train_loss)
                #     valid_loss_list.append(average_valid_loss)
                #     global_steps_list.append(global_step)
                #
                #     # resetting running values
                #     running_loss = 0.0
                #     valid_running_loss = 0.0
                #     model.train()
                #
                #     # print progress
                #     logger.info(f'Epoch [{epoch + 1}/{self.config.n_epochs}], '
                #                 f'Step [{global_step}/{self.config.n_epochs * len(train_iter)}], '
                #                 f'Train Loss: {average_train_loss}, Valid Loss: {average_valid_loss}')
                #
                #     # checkpoint
                #     if best_valid_loss > average_valid_loss:
                #         best_valid_loss = average_valid_loss
                #         self.save_checkpoint(model, model_name, best_valid_loss)

        self.save_checkpoint(model, model_name, 1.0)
        logger.info(f"Finished Training {model_name}!")
        return model

    def train(self, dump=True):
        logger.info(
            f"Training Dialogue Act Tagger for {self.config.taxonomy} taxonomy, using the following corpora:"
            f"{[c.name for c in self.corpora]}"
        )
        dataset = []
        for corpus in self.corpora:
            logger.info(f"Loading corpus {corpus.name}")
            dataset = dataset + corpus.utterances["train"]
        logger.info("All corpora loaded!")

        # Shuffle dataset and create train-validation split

        # TODO implement balanced split & shuffle instead of this
        random.shuffle(dataset)
        random.shuffle(dataset)
        random.shuffle(dataset)

        # TODO remove this, it's only here for debugging
        dataset = dataset[:600]

        train_set = dataset[: int(len(dataset) * 0.85)]
        valid_set = dataset[int(len(dataset) * 0.85) + 1 :]

        models = {}
        if "dimension" in self.config.taxonomy.value.__annotations__.keys():
            # Train dimension tagger
            logger.info("Training dimension pipeline")
            dimension_train = stringify_tags(train_set, "dimension")
            dimension_dev = stringify_tags(valid_set, "dimension")

            dimension_values = list(
                self.config.taxonomy.value.get_dimension_taxonomy().values().keys()
            )
            print(dimension_values)
            models["dimension"] = self.train_transformer(
                dimension_train,
                dimension_dev,
                len(dimension_values),
                model_name="dimension",
            )

            for dimension_value in dimension_values:
                logger.info(
                    f"Training communication function pipeline for dimension {dimension_value}"
                )
                comm_train = stringify_tags(
                    train_set,
                    "comm_function",
                    filter_attr="dimension",
                    filter_value=dimension_value,
                )
                comm_dev = stringify_tags(
                    valid_set,
                    "comm_function",
                    filter_attr="dimension",
                    filter_value=dimension_value,
                )
                comm_labels = [[tag for tag in utt.tags] for utt in comm_train]

                comm_values = list(
                    set([label for tagset in comm_labels for label in tagset])
                )
                models[f"comm_{dimension_value}"] = self.train_transformer(
                    comm_train,
                    comm_dev,
                    len(comm_values),
                    model_name=f"comm_{dimension_value}",
                )
        else:
            logger.info("Training unified communication function pipeline")
            comm_train = stringify_tags(train_set, "comm_function")
            comm_dev = stringify_tags(valid_set, "comm_function")
            comm_values = list(set([tag for utt in comm_train for tag in utt.tags]))
            models["comm_all"] = self.train_transformer(
                comm_train, comm_dev, len(comm_values), model_name="comm_all"
            )
        self.config.pipelines = list(models.keys())
        return TransformerTagger(self.config)
