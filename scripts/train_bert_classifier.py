import os
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertModel, BertForTokenClassification, BertAdam
from diactag.data import TokenDataLoader, SentDataLoader
from diactag.configs import BertConfig
from diactag.metrics import flat_accuracy
from tqdm import tqdm, trange
import numpy as np
from seqeval.metrics import f1_score

def main():

    bert_config = BertConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.get_device_name(0)
    
    print("Loading padded, tokenised data")

    input_ids, tags, tags_vals, tag2idx = diactag.corpus.data(bert_config.data_spec)

    print("splitting into folds and ")

    attention_masks = [[float(i>0) for i in ii] for ii in input_ids]
    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, 
                                                                random_state=2018, test_size=0.1)
    tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                random_state=2018, test_size=0.1)
    tr_inputs = torch.tensor(tr_inputs)
    val_inputs = torch.tensor(val_inputs)
    tr_tags = torch.tensor(tr_tags)
    val_tags = torch.tensor(val_tags)
    tr_masks = torch.tensor(tr_masks)
    val_masks = torch.tensor(val_masks)

    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                            bert_config.batch_size)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler,
                            bert_config.batch_size)

    if bert_config.output_type == "sentence":
        train_dl, valid_dl, test_dl = SentDataLoader(bert_config)
        # TODO: model = BertModel()
    elif bert_config.output_type == "token":
        train_dl, valid_dl, test_dl = TokenDataLoader(bert_config)
        model = BertForTokenClassification.from_pretrained(
            bert_config,
            num_labels=len(bert_config.tag2idx)
            )
    else:
        raise ValueError("Invalid model output prediction type")

    # DON'T WORRY ABOUT LABEL EMBEDDINGS FOR NOW
    # if slrr_config.pretrain_embs:
    #     print("Initialising EmbNet model")
    #     emb_net = LabelEmbNet(slrr_config.emb_config, rr_dl)
    #     print("Initialising EmbNet trainer")
    #     emb_trainer = LabelTrainer(sess, emb_net, rr_dl)
    #     print("Beginning training...")
    #     emb_trainer.train()
    #     print("Saving pre-trained label embeddings...")
    #     emb_trainer.save_np_embeddings()
    #     print("Done!")

    # pass parameters to GPU if available
    if (torch.cuda.is_available() and bert_config.device=='gpu'):
        model.cuda()

    if bert_config.fine_tuning == "full":
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': bert_config.weight_decay_rate},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters()) 
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = Adam(optimizer_grouped_parameters, lr=bert_config.learning_rate)

    print("Beginning training...")
    for _ in trange(bert_config.n_epochs, desc="Epoch"):
        # TRAIN loop
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for _, batch in enumerate(train_dl):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # forward pass
            loss = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
            # backward pass
            loss.backward()
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                    max_norm=bert_config.clip_grad_norm)
            # update parameters
            optimizer.step()
            model.zero_grad()
        # print train loss per epoch
        print("Train loss: {}".format(tr_loss/nb_tr_steps))
        # save model and optimiser state
        torch.save(model.state_dict(), os.path.join(bert_config.save_path, 'model.pth'))
        torch.save(optimizer.state_dict(), os.path.join(bert_config.save_path, 'optimiser.pth'))
        # VALIDATION on validation set
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], []
        for batch in valid_dl:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
            with torch.no_grad():
                tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                    attention_mask=b_input_mask, labels=b_labels)
                logits = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)
            
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            
            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
        pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
        valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
        print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))

    print("Done!")

if __name__ == "__main__":
    main()


