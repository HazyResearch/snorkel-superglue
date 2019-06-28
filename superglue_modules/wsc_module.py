import torch
from torch import nn
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor


class SpanClassifierModule(nn.Module):
    def _make_span_extractor(self):
        return SelfAttentiveSpanExtractor(self.proj_dim)

    def _make_cnn_layer(self, d_inp):
        """
        Make a CNN layer as a projection of local context.
        CNN maps [batch_size, max_len, d_inp]
        to [batch_size, max_len, proj_dim] with no change in length.
        """
        k = 1 + 2 * self.cnn_context
        padding = self.cnn_context
        return nn.Conv1d(
            d_inp,
            self.proj_dim,
            kernel_size=k,
            stride=1,
            padding=padding,
            dilation=1,
            groups=1,
            bias=True,
        )

    def __init__(
        self,
        d_inp=1024,
        proj_dim=512,
        num_spans=2,
        cnn_context=0,
        n_classes=2,
        dropout=0.1,
    ):
        super().__init__()

        self.cnn_context = cnn_context
        self.num_spans = num_spans
        self.proj_dim = proj_dim
        self.dropout = nn.Dropout(dropout)

        self.projs = torch.nn.ModuleList()

        for i in range(num_spans):
            # create a word-level pooling layer operator
            proj = self._make_cnn_layer(d_inp)
            self.projs.append(proj)
        self.span_extractors = torch.nn.ModuleList()

        # Lee's self-pooling operator (https://arxiv.org/abs/1812.10860)
        for i in range(num_spans):
            span_extractor = self._make_span_extractor()
            self.span_extractors.append(span_extractor)

        # Classifier gets concatenated projections of spans.
        clf_input_dim = self.span_extractors[1].get_output_dim() * num_spans
        self.classifier = nn.Linear(clf_input_dim, n_classes)

    def forward(self, feature, span1_idxs, span2_idxs, mask):
        # Apply projection CNN layer for each span of the input sentence
        sent_embs_t = self.dropout(feature[-1]).transpose(1, 2)  # needed for CNN layer

        se_projs = []
        for i in range(self.num_spans):
            se_proj = self.projs[i](sent_embs_t).transpose(2, 1).contiguous()
            se_projs.append(se_proj)

        span_embs = None

        _kw = dict(sequence_mask=mask.unsqueeze(2).long())
        span_idxs = [span1_idxs.unsqueeze(1), span2_idxs.unsqueeze(1)]
        for i in range(self.num_spans):
            # spans are [batch_size, num_targets, span_modules]
            span_emb = self.span_extractors[i](se_projs[i], span_idxs[i], **_kw)
            if span_embs is None:
                span_embs = span_emb
            else:
                span_embs = torch.cat([span_embs, span_emb], dim=2)

        # [batch_size, num_targets, n_classes]
        logits = self.classifier(span_embs).squeeze(1)

        return logits
