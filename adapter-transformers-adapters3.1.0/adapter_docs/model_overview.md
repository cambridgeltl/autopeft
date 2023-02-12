# Model Overview

This page gives an overview of the Transformer models currently supported by `adapter-transformers`.
The table below further shows which model architectures support which adaptation methods and which features of `adapter-transformers`.

```{eval-rst}
.. note::
    Each supported model architecture X typically provides a class ``XAdapterModel`` for usage with ``AutoAdapterModel``.
    Additionally, it is possible to use adapters with the model classes already shipped with HuggingFace Transformers.
    E.g., for BERT, this means adapter-transformers provides a ``BertAdapterModel`` class, but you can also use ``BertModel``, ``BertForSequenceClassification`` etc. together with adapters.
```

| Model                                   | (Bottleneck)<br> Adapters | Prefix<br> Tuning | LoRA | Compacter | Adapter<br> Fusion | Invertible<br> Adapters | Parallel<br> block |
| --------------------------------------- | -| - | - | - | - | - | - |
| [BART](classes/models/bart.html)        | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [BERT](classes/models/bert.html)        | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [DeBERTa](classes/models/deberta.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [DeBERTa-v2](classes/models/debertaV2.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [DistilBERT](classes/models/distilbert.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [Encoder Decoder](classes/models/encoderdecoder.html) | (*) | (*) | (*) | (*) | (*) | (*) | |
| [GPT-2](classes/models/gpt2.html)       | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [MBart](classes/models/mbart.html)      | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [RoBERTa](classes/models/roberta.html)  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [T5](classes/models/t5.html)            | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [ViT](classes/models/vit.html)            | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [XLM-RoBERTa](classes/models/xlmroberta.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

(*) If the used encoder and decoder model class are supported.

**Missing a model architecture you'd like to use?**
adapter-transformers can be easily extended to new model architectures as described in [Adding Adapters to a Model](https://docs.adapterhub.ml/contributing/adding_adapters_to_a_model.html).
Feel free to [open an issue](https://github.com/Adapter-Hub/adapter-transformers/issues) requesting support for a new architecture.
_We very much welcome pull requests adding new model implementations!_
