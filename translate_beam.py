import os
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.serialization import default_restore_location

from seq2seq import models, utils
from seq2seq.data.dictionary import Dictionary
from seq2seq.data.dataset import Seq2SeqDataset, BatchSampler
from seq2seq.beam import BeamSearch, BeamSearchNode


def get_args():
    """ Defines generation-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Sequence to Sequence Model')
    parser.add_argument('--cuda', default=False, help='Use a GPU')
    parser.add_argument('--seed', default=42, type=int, help='pseudo random number generator seed')

    # Add data arguments
    parser.add_argument('--data', default='data_asg4/prepared_data', help='path to data directory')
    parser.add_argument('--checkpoint-path', default='checkpoints_asg4/checkpoint_best.pt', help='path to the model file')
    parser.add_argument('--batch-size', default=None, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--output', default='model_translations.txt', type=str,
                        help='path to the output file destination')
    parser.add_argument('--max-len', default=100, type=int, help='maximum length of generated sequence')

    # Add beam search arguments
    parser.add_argument('--beam-size', default=5, type=int, help='number of hypotheses expanded in beam search')

    return parser.parse_args()


def main(args):
    """ Main translation function' """
    # Load arguments from checkpoint
    torch.manual_seed(args.seed)
    state_dict = torch.load(args.checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    args_loaded = argparse.Namespace(**{**vars(args), **vars(state_dict['args'])})
    args_loaded.data = args.data
    args = args_loaded
    utils.init_logging(args)

    # Load dictionaries
    src_dict = Dictionary.load(os.path.join(args.data, 'dict.{:s}'.format(args.source_lang)))
    logging.info('Loaded a source dictionary ({:s}) with {:d} words'.format(args.source_lang, len(src_dict)))
    tgt_dict = Dictionary.load(os.path.join(args.data, 'dict.{:s}'.format(args.target_lang)))
    logging.info('Loaded a target dictionary ({:s}) with {:d} words'.format(args.target_lang, len(tgt_dict)))

    # Load dataset
    test_dataset = Seq2SeqDataset(
        src_file=os.path.join(args.data, 'test.{:s}'.format(args.source_lang)),
        tgt_file=os.path.join(args.data, 'test.{:s}'.format(args.target_lang)),
        src_dict=src_dict, tgt_dict=tgt_dict)

    test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=1, collate_fn=test_dataset.collater,
                                              batch_sampler=BatchSampler(test_dataset, 9999999,
                                                                         args.batch_size, 1, 0, shuffle=False,
                                                                         seed=args.seed))
    # Build a Seq2SeqModel model (args.arch = LSTM), and criterion
    model = models.build_model(args, src_dict, tgt_dict)
    if args.cuda:
        model = model.cuda()
    model.eval()
    model.load_state_dict(state_dict['model'])
    logging.info('Loaded a model from checkpoint {:s}'.format(args.checkpoint_path))
    progress_bar = tqdm(test_loader, desc='| Generation', leave=False)

    # Iterate over the test set
    all_hyps = {}
    for i, sample in enumerate(progress_bar):

        # Create a beam search object or every input sentence in batch
        # a) int, batch size
        batch_size = sample['src_tokens'].shape[0]

        # b) list of seq2seq.beam.BeamSearch objects, len=args.batch-size
        searches = [BeamSearch(beam_size=args.beam_size, max_len=args.max_len - 1, pad=tgt_dict.unk_idx) for i in range(batch_size)]
        
        with torch.no_grad():
            # Compute the encoder output, a dict with:
            # 'src_embeddings', [src_time_steps, batch_size, num_features], 
            # 'src_out', [lstm_output, final_hidden_states, final_cell_states], 
            # 'src_mask', for each batch, each sequence where pad_idx begins as True
            encoder_out = model.encoder(src_tokens=sample['src_tokens'], src_lengths=sample['src_lengths'])
            
            # __QUESTION 1: What is "go_slice" used for and what do its dimensions represent?
            # tensor 64x1 filled with 1s, used as target inputs
            go_slice = \
                torch.ones(sample['src_tokens'].shape[0], 1).fill_(tgt_dict.eos_idx).type_as(sample['src_tokens'])
                
            if args.cuda:
                go_slice = utils.move_to_cuda(go_slice)

            # Compute the decoder output at the first time step from (tgt_inputs, encoder_out)
            # return decoder_output, [batch_size, tgt_time_steps, num_features],  and attn_weights
            decoder_out, _ = model.decoder(tgt_inputs=go_slice, encoder_out=encoder_out)


            # __QUESTION 2: Why do we keep one top candidate more than the beam size?
            # return namedtuple of (values, indices), with:
            # log_probs - SM probabilities, torch.Size([64, 1, 3]), [batch-size, 1, beam_size+1] 
            # next_candidates - the indices of the elements in the original input tensor (inputs - translated tokens)
            # same size as log_probs.
            # Extra 1 - to compensate for <\s> or <unk> token.
            log_probs, next_candidates = torch.topk(input=torch.log(torch.softmax(decoder_out, dim=2)),
                                                    k=args.beam_size+1, dim=-1)

            
            
        # Create number of beam_size beam search nodes for every input sentence
        for i in range(batch_size):
            for j in range(args.beam_size):
                best_candidate = next_candidates[i, :, j]
                # what is backoff candidate? is used if best_cand is invalid (EOS or UNK)?
                backoff_candidate = next_candidates[i, :, j+1]
                best_log_p = log_probs[i, :, j]
                backoff_log_p = log_probs[i, :, j+1]
                next_word = torch.where(best_candidate == tgt_dict.unk_idx, backoff_candidate, best_candidate)
                log_p = torch.where(best_candidate == tgt_dict.unk_idx, backoff_log_p, best_log_p)
                log_p = log_p[-1]

                # Store the encoder_out information for the current input sentence and beam
                emb = encoder_out['src_embeddings'][:,i,:]
                lstm_out = encoder_out['src_out'][0][:,i,:]
                final_hidden = encoder_out['src_out'][1][:,i,:]
                final_cell = encoder_out['src_out'][2][:,i,:]
                try:
                    mask = encoder_out['src_mask'][i,:]
                except TypeError:
                    mask = None

                # for each batch item i, produce beam_size `node` objects
                # total: batch_size x beam_size 
                node = BeamSearchNode(searches[i], emb, lstm_out, final_hidden, final_cell,
                                      mask, sequence=torch.cat((go_slice[i], next_word)), 
                                      logProb=log_p, length=1
                                )
                # __QUESTION 3: Why do we add the node with a negative score?
                # Adds a new beam search node to the priority queue of current nodes
                # We're dealing with PriorityQueue data structure object, 
                # class PriorityQueue(Queue) - Variant of Queue that retrieves open entries in priority order (lowest first).
                # node.eval() returns log prob.
                searches[i].add(score=-node.eval(), node=node)

        # Start generating further tokens until max sentence length reached
        for _ in range(args.max_len-1):

            # Get the current nodes to expand
            nodes = [n[1] for s in searches for n in s.get_current_beams()]
            if nodes == []:
                break # All beams ended in EOS

            # Reconstruct prev_words, encoder_out from current beam search nodes
            prev_words = torch.stack([node.sequence for node in nodes])
            encoder_out["src_embeddings"] = torch.stack([node.emb for node in nodes], dim=1)
            lstm_out = torch.stack([node.lstm_out for node in nodes], dim=1)
            final_hidden = torch.stack([node.final_hidden for node in nodes], dim=1)
            final_cell = torch.stack([node.final_cell for node in nodes], dim=1)
            encoder_out["src_out"] = (lstm_out, final_hidden, final_cell)
            try:
                encoder_out["src_mask"] = torch.stack([node.mask for node in nodes], dim=0)
            except TypeError:
                encoder_out["src_mask"] = None

            with torch.no_grad():
                # Compute the decoder output by feeding it the decoded sentence prefix
                decoder_out, _ = model.decoder(prev_words, encoder_out)

            # see __QUESTION 2
            log_probs, next_candidates = torch.topk(torch.log(torch.softmax(decoder_out, dim=2)), args.beam_size+1, dim=-1)

            # Create number of beam_size next nodes for every current node
            for i in range(log_probs.shape[0]):
                for j in range(args.beam_size):

                    best_candidate = next_candidates[i, :, j]
                    backoff_candidate = next_candidates[i, :, j+1]
                    best_log_p = log_probs[i, :, j]
                    backoff_log_p = log_probs[i, :, j+1]
                    next_word = torch.where(best_candidate == tgt_dict.unk_idx, backoff_candidate, best_candidate)
                    log_p = torch.where(best_candidate == tgt_dict.unk_idx, backoff_log_p, best_log_p)
                    log_p = log_p[-1]
                    next_word = torch.cat((prev_words[i][1:], next_word[-1:]))

                    # Get parent node and beam search object for corresponding sentence
                    node = nodes[i]
                    search = node.search

                    # __QUESTION 4: How are "add" and "add_final" different? What would happen if we did not make this distinction?

                    # Store the node as final if EOS is generated
                    if next_word[-1 ] == tgt_dict.eos_idx:
                        node = BeamSearchNode(search, node.emb, node.lstm_out, node.final_hidden,
                                              node.final_cell, node.mask, torch.cat((prev_words[i][0].view([1]),
                                              next_word)), node.logp, node.length)
                        # Adds a beam search path that ended in EOS (= finished sentence)
                        search.add_final(-node.eval(), node)

                    # Add the node to current nodes for next iteration
                    else:
                        node = BeamSearchNode(search, node.emb, node.lstm_out, node.final_hidden,
                                              node.final_cell, node.mask, torch.cat((prev_words[i][0].view([1]),
                                              next_word)), node.logp + log_p, node.length + 1)
                        search.add(-node.eval(), node)

            # __QUESTION 5: What happens internally when we prune our beams?
            # How do we know we always maintain the best sequences?
            # Removes all nodes but the beam_size best ones (lowest neg log prob)
            for search in searches:
                search.prune()

        # Segment into sentences
        best_sents = torch.stack([search.get_best()[1].sequence[1:].cpu() for search in searches])
        decoded_batch = best_sents.numpy()

        output_sentences = [decoded_batch[row, :] for row in range(decoded_batch.shape[0])]

        # __QUESTION 6: What is the purpose of this for loop?
        # temp = output_sentences and contains batch_size items each with ids of vocabulary items up until EOS
        # print("output_sentences before: ", output_sentences)

        temp = list()
        for sent in output_sentences:
            first_eos = np.where(sent == tgt_dict.eos_idx)[0]
            if len(first_eos) > 0:
                temp.append(sent[:first_eos[0]])
            else:
                temp.append(sent)
        output_sentences = temp
        # print("output_sentences: ", output_sentences)

        # Convert arrays of indices into strings of words
        output_sentences = [tgt_dict.string(sent) for sent in output_sentences]

        for ii, sent in enumerate(output_sentences):
            all_hyps[int(sample['id'].data[ii])] = sent


    # Write to file
    if args.output is not None:
        with open(args.output, 'w') as out_file:
            for sent_id in range(len(all_hyps.keys())):
                out_file.write(all_hyps[sent_id] + '\n')


if __name__ == '__main__':
    args = get_args()
    main(args)
