# Base Python
from collections import OrderedDict

# Scientific modules
import numpy as np

import snorkel

# ddlite LSTM
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class LSTM(object):

  def __init__(self, candidates, training, lf_probs):
    self.C = candidates
    self.training = training
    self.lf_probs = lf_probs
    # LSTM
    self.lstm_SEED = 123
    self.lstm_params = None
    self.lstm_tparams = None
    self.lstm_settings = None
    self.lstm_X = None
    self.word_dict = None
    self.marginals = None

  def num_candidates(self):
    return len(self.C)

  def ortho_weight(self):
    u, s, v = np.linalg.svd(np.random.randn(self.lstm_settings['dim'], self.lstm_settings['dim']))
    return u.astype(config.floatX)

  def init_lstm_params(self):
    params = OrderedDict()
    # embedding
    randn = np.random.rand(self.lstm_settings['word_size'], self.lstm_settings['dim'])
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    # lstm
    params['lstm_W'] = np.concatenate([self.ortho_weight(), self.ortho_weight(), self.ortho_weight(), self.ortho_weight()], axis = 1)
    params['lstm_U'] = np.concatenate([self.ortho_weight(), self.ortho_weight(), self.ortho_weight(), self.ortho_weight()], axis = 1)
    params['lstm_b'] = np.zeros((4 * self.lstm_settings['dim'],)).astype(config.floatX)
    # classifier
    params['U'] = 0.01 * np.random.randn(self.lstm_settings['dim'], self.lstm_settings['label_dim']).astype(config.floatX)
    params['b'] = np.zeros((self.lstm_settings['label_dim'],)).astype(config.floatX)
    self.lstm_params=params

  def init_lstm_theano_params(self):
    tparams = OrderedDict()
    for k, v in self.lstm_params.items():
        tparams[k] = theano.shared(self.lstm_params[k], name = k)
    self.lstm_tparams=tparams

  def dropout_layer(self, state_before, use_noise, rnd):
    proj = tensor.switch(use_noise,
                         (state_before * rnd.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj

  def adadelta(lr, tparams, grads, x, mask, y, w, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * np.asarray(0., dtype = config.floatX), name='%s_grad' % k) for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * np.asarray(0., dtype = config.floatX), name='%s_rup2' % k) for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * np.asarray(0., dtype = config.floatX), name='%s_rgrad2' % k) for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y, w], cost, updates=zgup + rg2up, name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up, on_unused_input='ignore', name='adadelta_f_update')
    return f_grad_shared, f_update

  def build_lstm(self):
    def _slice(_x, n, dim):
      if _x.ndim  == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
      else:
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
      preact = tensor.dot(h_, self.lstm_tparams['lstm_U'])
      preact += x_
      i = tensor.nnet.sigmoid(_slice(preact, 0, self.lstm_settings['dim']))
      f = tensor.nnet.sigmoid(_slice(preact, 1, self.lstm_settings['dim']))
      o = tensor.nnet.sigmoid(_slice(preact, 2, self.lstm_settings['dim']))
      c = tensor.tanh(_slice(preact, 3, self.lstm_settings['dim']))

      c = f * c_ + i * c
      c = m_[:, None] * c + (1. - m_)[:, None] * c_
      h = o * tensor.tanh(c)
      h = m_[:, None] * h + (1. - m_)[:, None] * h_
      return h, c

    rnd = RandomStreams(self.lstm_SEED)
    # Used for dropout.
    use_noise = theano.shared(np.asarray(0., dtype = config.floatX))

    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')
    w = tensor.vector('w', dtype=config.floatX)

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]
    emb = self.lstm_tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, self.lstm_settings['dim']])

    state_below = (tensor.dot(emb, self.lstm_tparams['lstm_W']) + self.lstm_tparams['lstm_b'])
    rval, updates = theano.scan(_step,
                                sequences = [mask, state_below],
                                outputs_info = [tensor.alloc(np.asarray(0., dtype = config.floatX),
                                                             n_samples,
                                                             self.lstm_settings['dim']),
                                                tensor.alloc(np.asarray(0., dtype = config.floatX),
                                                             n_samples,
                                                             self.lstm_settings['dim'])],
                                name = 'lstm_layers',
                                n_steps = n_timesteps)

    proj = rval[0]
    proj = (proj * mask[:, :, None]).sum(axis=0)
    proj = proj / mask.sum(axis=0)[:, None]

    if self.lstm_settings['dropout']:
      proj = self.dropout_layer(proj, use_noise, rnd)

    pred = tensor.nnet.softmax(tensor.dot(proj, self.lstm_tparams['U']) + self.lstm_tparams['b'])

    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    eps = 1e-6

    cost = -(w * tensor.log(pred[tensor.arange(n_samples), y] + eps) + (1. - w) * tensor.log(1. - pred[tensor.arange(n_samples), y] + eps)).mean()
    return use_noise, x, mask, y, f_pred_prob, f_pred, cost, w

  def mini_batches(self, n, batch_size, shuffle = False):
    ids = np.arange(n, dtype = "int32")
    if shuffle:
      np.random.shuffle(ids)
    minibatches = []
    start = 0
    for i in range(n // batch_size):
      minibatches.append(ids[start: start + batch_size])
      start = start + batch_size
    if start != n:
      minibatches.append(ids[start:])
    return zip(range(len(minibatches)), minibatches)

  def process_data(self, x, y, w, maxlen = None):
    lengths = [len(i) for i in x]

    # filter all examples which length > maxlen
    if maxlen is not None:
      _x, _y, _w, _lengths = [], [], [], []
      for __x, __y, __l, __w in zip(x, y, lengths, w):
        if __l <= maxlen:
          _x.append(__x)
          _y.append(__y)
          _lengths.append(__l)
          _w.append(__w)
      lengths = _lengths
      x = _x
      y = _y
      w = _w
      if len(lengths) == 0:
        return None, None, None

    nsamples = len(x)
    maxlen = np.max(lengths)
    _x = np.zeros((maxlen, nsamples)).astype('int64')
    _x_mask = np.zeros((maxlen, nsamples)).astype(theano.config.floatX)

    for idx, sample in enumerate(x):
      _x[:lengths[idx], idx] = sample
      _x_mask[:lengths[idx], idx] = 1.

    return _x, _x_mask, y, w

  def transfer_params_from_gpu_to_cpy(self):
    params = OrderedDict()
    for k, v in self.lstm_tparams.items():
      params[k] = v.get_value()
    return params

  def pred(self, f_pred, data, minibatches):
    error = 0
    for id, samples in minibatches:
      x = [data[1][i] for i in samples]
      y = [data[2][i] for i in samples]
      w = [data[3][i] for i in samples]
      x, mask, y, w = self.process_data(x, y, w, maxlen = None)
      preds = f_pred(x, mask)
      error += (preds == y).sum()
    error = 1. - np.asarray(error, dtype = config.floatX) / len(data[0])
    return error

  def pred_p(self, f_pred_prob, data, minibatches):
    error = 0
    res = []
    for id, samples in minibatches:
      l = [data[0][i] for i in samples]
      x = [data[1][i] for i in samples]
      y = [data[2][i] for i in samples]
      w = [data[3][i] for i in samples]
      x, mask, y, w = self.process_data(x, y, w, maxlen = None)
      preds = f_pred_prob(x, mask)
      pred_l = preds.argmax(axis = 1)
      res.extend([(a, b[1]) for a, b in zip(l, preds)])
    return res

  def lstm(self,
           dim = 50,                # word embedding dimension
           batch_size = 100,        # batch size
           learning_rate = 0.01,    # learning rate for sgd
           optimizer = adadelta,    # optimizer
           epoch = 300,             # learning epoch
           maxlen = 1000,           # max sequence len
           dropout = True,
           verbose = True):

    self.lstm_settings = locals().copy()

    self.lstm_settings['label_dim'] = 2
    self.lstm_settings['word_size'] = len(self.word_dict)
    self.init_lstm_params()

    self.init_lstm_theano_params()

    use_noise, x, mask, y, f_pred_prob, f_pred, cost, w = self.build_lstm()

    f_cost = theano.function([x, mask, y, w], cost, name='f_cost')
    grads = tensor.grad(cost, wrt=list(self.lstm_tparams.values()))
    f_grad = theano.function([x, mask, y, w], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, self.lstm_tparams, grads, x, mask, y, w, cost)

    train_data, x, y, w = [], [], [], []
    for idx in self.training:
        x.append(self.lstm_X[idx])
        if self.lf_probs[idx]>0.5:
          y.append(1)
          w.append(self.lf_probs[idx])
        else:
          y.append(0)
          w.append(1.-self.lf_probs[idx])
    train_data=[self.training,x,y,w]

    test_data=[range(self.num_candidates()),self.lstm_X,[1]*self.num_candidates(),[1.]*self.num_candidates()]

    error_log = []
    for idx in range(epoch):
      ids = self.mini_batches(len(train_data[0]), batch_size, shuffle=True)
      for id, samples in ids:
        use_noise.set_value(1.)
        x = [train_data[1][i] for i in samples]
        y = [train_data[2][i] for i in samples]
        w = np.array([train_data[3][i] for i in samples]).astype(config.floatX)
        x, mask, y, w = self.process_data(x, y, w, maxlen)
        preds = f_pred_prob(x, mask)
        cost = f_grad_shared(x, mask, y, w)
        f_update(learning_rate)
        if np.isnan(cost) or np.isinf(cost):
          raise ValueError("Bad cost")
      use_noise.set_value(0.)
      train_error = self.pred(f_pred, train_data, ids)
      if error_log == [] or train_error < min(error_log):
        self.lstm_params = self.transfer_params_from_gpu_to_cpy()
      if verbose:
        print ("Epoch #%d, Training error: %f") % (idx, train_error)
    use_noise.set_value(0.)
    ids = self.mini_batches(len(test_data[0]), batch_size, shuffle=True)
    pred=self.pred_p(f_pred_prob, test_data, ids)
    self.marginals = np.array([0.]*len(pred))
    for id, p in pred:
      self.marginals[id]=p

  def get_word_dict(self, contain_mention, word_window_length, ignore_case):
    """
    Get array of training word sequences
    Return word dictionary
    """
    lstm_dict = {'__place_holder__':0, '__unknown__':1}
    if type(self.C) is snorkel.Entities:
      words=[]
      for i in self.training:
        min_idx=min(self.C._candidates[i].idxs)
        max_idx=max(self.C._candidates[i].idxs)+1
        length=len(self.C._candidates[i].words)
        lw= range(max(0,min_idx-word_window_length), min_idx)
        rw= range(max_idx,min(max_idx+word_window_length, length))
        m=self.C._candidates[i].idxs
        w=np.array(self.C._candidates[i].words)
        m=w[m] if contain_mention else []
        seq = [_.lower() if ignore_case else _ for _ in np.concatenate((w[lw],m,w[rw]))]
        words +=seq
      words = list(set(words))
      for i in range(len(words)):
        lstm_dict[words[i]]=i+2
      self.word_dict=lstm_dict
    if type(self.C) is snorkel.Relations:
      words=[]
      for i in self.training:
        e1_idxs=self.C._candidates[i].e1_idxs
        e2_idxs=self.C._candidates[i].e2_idxs
        if (int(min(e1_idxs))>int(max(e2_idxs)) or int(min(e2_idxs))>int(max(e1_idxs))):
          pass
        else:
          continue
        start_idx = min(int(max(e1_idxs)), int(max(e2_idxs)))+1
        end_idx = max(int(min(e1_idxs)), int(min(e2_idxs)))
        min_idx=min(min(e1_idxs), min(e2_idxs))
        max_idx=max(max(e1_idxs), max(e2_idxs))+1
        length=len(self.C._candidates[i].words)
        lw=range(max(0,min_idx-word_window_length), min_idx)
        rw=range(max_idx,min(max_idx+word_window_length, length))
        w=np.array(self.C._candidates[i].words)
        if contain_mention:
          if min(e1_idxs) < min(e2_idxs):
            m1,m2=w[e1_idxs],w[e2_idxs]
          else:
            m1,m2=w[e2_idxs],w[e1_idxs]
        else:
          m1,m2=[],[]
        wq=self.C._candidates[i].words[start_idx:end_idx]
        seq = [_.lower() if ignore_case else _ for _ in np.concatenate((w[lw],m1,wq,m2,w[rw]))]
        words+=seq
      words=list(set(words))
      for i in range(len(words)):
        lstm_dict[words[i]]=i+2
      self.word_dict=lstm_dict

  def map_word_to_id(self, contain_mention, word_window_length, ignore_case):
    """
    Get array of candidate word sequences given word dictionary
    Return array of candidate id sequences
    """
    self.lstm_X=[]
    if type(self.C) is snorkel.Entities:
      words=[]
      for i in range(self.num_candidates()):
        min_idx=min(self.C._candidates[i].idxs)
        max_idx=max(self.C._candidates[i].idxs)+1
        length=len(self.C._candidates[i].words)
        lw= range(max(0,min_idx-word_window_length), min_idx)
        rw= range(max_idx,min(max_idx+word_window_length, length))
        m=self.C._candidates[i].idxs
        w=np.array(self.C._candidates[i].words)
        m=w[m] if contain_mention else ['__place_holder__']
        seq = [_.lower() if ignore_case else _ for _ in np.concatenate((w[lw],m,w[rw]))]
        x=[0]+[self.word_dict[j] if j in self.word_dict else 1 for j in seq]+[0]
        self.lstm_X.append(x)
    if type(self.C) is snorkel.Relations:
      words=[]
      for i in range(self.num_candidates()):
        e1_idxs=self.C._candidates[i].e1_idxs
        e2_idxs=self.C._candidates[i].e2_idxs
        if (int(min(e1_idxs))>int(max(e2_idxs)) or int(min(e2_idxs))>int(max(e1_idxs))):
          pass
        else:
          self.lstm_X.append([0])
          continue
        start_idx = min(int(max(e1_idxs)), int(max(e2_idxs)))+1
        end_idx = max(int(min(e1_idxs)), int(min(e2_idxs)))
        min_idx=min(min(e1_idxs), min(e2_idxs))
        max_idx=max(max(e1_idxs), max(e2_idxs))+1
        length=len(self.C._candidates[i].words)
        lw=range(max(0,min_idx-word_window_length), min_idx)
        rw=range(max_idx,min(max_idx+word_window_length, length))
        w=np.array(self.C._candidates[i].words)
        if contain_mention:
          if min(e1_idxs) < min(e2_idxs):
            m1,m2=w[e1_idxs],w[e2_idxs]
          else:
            m1,m2=w[e2_idxs],w[e1_idxs]
        else:
          m1,m2=['__place_holder__'],['__place_holder__']
        wq=self.C._candidates[i].words[start_idx:end_idx]
        seq = [_.lower() if ignore_case else _ for _ in np.concatenate((w[lw],m1,wq,m2,w[rw]))]
        x=[0]+[self.word_dict[j] if j in self.word_dict else 1 for j in seq]+[0]
        self.lstm_X.append(x)

  def train_model_lstm(self, **kwargs):
    contain_mention=kwargs.get('contain_mention', True)
    word_window_length=kwargs.get('word_window_length', 0)
    ignore_case=kwargs.get('ignore_case', True)

    dim = kwargs.get('dim', 50)
    batch_size = kwargs.get('batch_size', 100)
    learning_rate = kwargs.get('rate', 0.01)
    epoch = kwargs.get('n_iter', 300)
    maxlen = kwargs.get('maxlen', 100)
    dropout = kwargs.get('dropout', True)
    verbose=kwargs.get('verbose', False)

    self.get_word_dict(contain_mention=contain_mention, word_window_length=word_window_length, ignore_case=ignore_case)
    self.map_word_to_id(contain_mention=contain_mention, word_window_length=word_window_length, ignore_case=ignore_case)
    self.lstm(dim=dim, batch_size=batch_size, learning_rate=learning_rate, epoch=epoch, dropout=dropout, verbose=verbose, maxlen=maxlen)

