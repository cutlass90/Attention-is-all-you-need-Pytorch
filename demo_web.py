import torch
from transformer.transformer import Transformer
from tqdm import tqdm
import streamlit as st

st.set_page_config(layout="wide")

class Inferencer:
    def __init__(self, ckpt_path, device):
        self.device = device
        state_dict = torch.load(ckpt_path)
        opt = state_dict['settings']
        self.net = Transformer(opt.src_vocab_size, opt.trg_vocab_size, opt.src_pad_idx, opt.trg_pad_idx,
                          opt.d_word_vec, opt.n_head, opt.d_model, opt.d_k, opt.d_v, opt.n_layers, 200, opt.embs_share_weight).to(device).eval()
        self.net.load_state_dict(state_dict['model'])
        self.src_vocab = state_dict['src_vocab']
        self.trg_vocab = state_dict['trg_vocab']

    def translate_sentence(self, sent: str)->str:
        src_seq = sent.split(' ')
        src_seq = [self.src_vocab.stoi.get(word, self.src_vocab.stoi[self.src_vocab.UNK]) for word in src_seq]
        src_seq = torch.LongTensor([src_seq]).to(self.device)
        pred = self.decode(src_seq)[0][1:]
        pred_sent = [self.trg_vocab.itos[ind] for ind in pred]
        return ' '.join(pred_sent)

    def decode(self, src_seq):
        trg_seq = torch.tensor([[2]]).to(src_seq.device)
        trg_seq = torch.cat([trg_seq for _ in range(src_seq.shape[0])])
        for step in tqdm(range(0, 50)):
            with torch.no_grad():
                pred = self.net(src_seq, trg_seq).view(src_seq.shape[0], trg_seq.shape[1], -1)
                trg_seq = torch.concatenate([trg_seq, pred.argmax(2)[:, step:step+1]], dim=1)
        return trg_seq







def main(ckpt_path, device):
    st.header("DE - EN translation demo")
    if not st.session_state.get('pipeline'):
        with st.spinner('loading models, wait...'):
            st.session_state['pipeline'] = Inferencer(ckpt_path, device)

    col1, col2 = st.columns(2)
    with col1:
        de_sentence = st.text_input('Enter De text here, each token separate with space', 'eine mutter und ihr kleiner sohn genießen einen schönen tag im freien .')



    if st.button('Translate'):
        with st.spinner('working, wait a moment'):
            translation = st.session_state['pipeline'].translate_sentence(de_sentence)
        with col2:
            st.text(translation)



if __name__ == "__main__":
    main(ckpt_path='weights/model_with_vocab.ckpt',
         device='cuda:0')