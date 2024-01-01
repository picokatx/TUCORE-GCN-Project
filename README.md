Bongseok Lee and Yong Suk Choi. Graph based network with contextualized representations of
turns in dialogue. In Proceedings of the 2021 Conference on Empirical Methods in Natural
Language Processing, pp. 443–455, 2021.

https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html


Issues:
- Entities are cast to [UNK] if not understood by the vocab: Replace names with special entity tokens
- Conversations are limited to 512 tokens
Improvements:
- Can replace GCN with Graph Transformer.
- Maybe design something that can distill multiple ongoing conversations at once?
https://arxiv.org/pdf/2305.00262.pdf HiDialog
- Replaced GCN with Graph Transformer Network
- Improved turn masking

https://arxiv.org/abs/2209.09146
https://aclanthology.org/2023.findings-emnlp.678.pdf
https://arxiv.org/abs/2108.13811
https://dl.acm.org/doi/abs/10.1145/3638760
https://dl.acm.org/doi/abs/10.1145/3616493
https://arxiv.org/abs/2306.06141
https://arxiv.org/abs/2310.14614