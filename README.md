Bongseok Lee and Yong Suk Choi. Graph based network with contextualized representations of
turns in dialogue. In Proceedings of the 2021 Conference on Empirical Methods in Natural
Language Processing, pp. 443–455, 2021.

https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html


Issues:
- Entities are cast to [UNK] if not understood by the vocab: Replace names with special entity tokens
- Conversations are limited to 512 tokens
- Limited to 9 Speakers
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


need to rerun data parser, forgot to add label ids the first time
bad accuracy on the first model is prob because I trained with batches of 32 but didn't adjust learning rate to account for this. need to implement scheduler from pytorch



PS D:\projects\affect\TUCORE-GCN> & d:/projects/affect/TUCORE-GCN/.venv/Scripts/Activate.ps1
(.venv) PS D:\projects\affect\TUCORE-GCN> python tucore_gcn_transformers/tucore_gcn_bert_train.py --do_train --do_eval --output_dir tucore_gcn_bert_test1
D:\projects\affect\TUCORE-GCN\.venv\Lib\site-packages\dgl\dgl.dll
01/03/2024 23:21:57 - INFO - __main__ -   device cuda n_gpu 1 distributed training False
01/03/2024 23:22:00 - INFO - __main__ -   ***** Running training *****
01/03/2024 23:22:00 - INFO - __main__ -     Num examples = 5997
01/03/2024 23:22:00 - INFO - __main__ -     Batch size = 12
01/03/2024 23:22:00 - INFO - __main__ -     Num steps = 4997
Epoch:   0%|                                                                                                                                                                                           | 0/20 [00:00<?, ?it/s]STAGE:2024-01-03 23:22:01 27064:22452 ..\third_party\kineto\libkineto\src\ActivityProfilerController.cpp:311] Completed Stage: Warm Up                                                                 | 0/499 [00:00<?, ?it/s]
We strongly recommend passing in an `attention_mask` since your input_ids may be padded. See https://huggingface.co/docs/transformers/troubleshooting#incorrect-output-when-padding-tokens-arent-masked.
STAGE:2024-01-03 23:22:07 27064:22452 ..\third_party\kineto\libkineto\src\ActivityProfilerController.cpp:317] Completed Stage: Collection
STAGE:2024-01-03 23:22:07 27064:22452 ..\third_party\kineto\libkineto\src\ActivityProfilerController.cpp:321] Completed Stage: Post Processing
------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                      Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                               aten::clone         0.01%     762.000us         1.16%      71.173ms       1.078ms     868.00 Kb      56.00 Kb       1.17 Gb     -20.00 Mb            66  
                          aten::contiguous         0.01%     515.000us         0.07%       4.045ms     155.577us     868.00 Kb      80.00 Kb     260.00 Mb      20.00 Mb            26  
                          aten::empty_like         0.01%     591.000us         2.73%     167.651ms     825.867us     812.00 Kb      52.00 Kb       4.29 Gb      45.92 Mb           203  
                               aten::empty         1.25%      76.857ms         1.25%      76.857ms     165.998us     760.41 Kb     760.41 Kb       4.51 Gb       4.51 Gb           463  
                              aten::arange         0.09%       5.747ms         0.18%      10.810ms     284.474us       3.39 Kb         496 b      24.00 Kb           0 b            38  
                                 aten::add         0.07%       4.590ms         0.07%       4.590ms      26.842us       1.70 Kb       1.70 Kb       2.32 Gb       2.32 Gb           171  
                             aten::resize_         0.01%     407.000us         0.01%     407.000us      11.306us       1.37 Kb       1.37 Kb      62.34 Mb      62.34 Mb            36  
                                  aten::to         0.01%     567.000us         0.24%      14.928ms      52.936us         416 b           0 b       1.30 Mb       3.50 Kb           282  
                            aten::_to_copy         0.01%     825.000us         0.24%      14.573ms     163.742us         416 b     -52.00 Kb       1.30 Mb           0 b            89  
                       aten::empty_strided         1.62%      99.491ms         1.62%      99.491ms     440.226us         416 b         416 b       3.11 Gb       3.11 Gb           226
------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 6.143s

                                                                                                                                                                                                                              STAGE:2024-01-03 23:22:26 27064:22452 ..\third_party\kineto\libkineto\src\ActivityProfilerController.cpp:311] Completed Stage: Warm Up                                                       | 1/499 [00:24<3:26:31, 24.88s/it] 
STAGE:2024-01-03 23:22:31 27064:22452 ..\third_party\kineto\libkineto\src\ActivityProfilerController.cpp:317] Completed Stage: Collection
STAGE:2024-01-03 23:22:31 27064:22452 ..\third_party\kineto\libkineto\src\ActivityProfilerController.cpp:321] Completed Stage: Post Processing
------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                      Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls
------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                          aten::empty_like         0.01%     336.000us         0.03%       1.336ms       6.581us     908.00 Kb     132.00 Kb       4.29 Gb     109.41 Mb           203
                               aten::clone         0.01%     267.000us         0.03%       1.438ms      21.788us     908.00 Kb    -100.00 Kb       1.16 Gb     -20.00 Mb            66
                          aten::contiguous         0.00%     194.000us         0.01%     540.000us      20.769us     908.00 Kb     456.00 Kb     258.50 Mb      19.50 Mb            26
                               aten::empty         0.03%       1.224ms         0.03%       1.224ms       2.655us     876.41 Kb     876.41 Kb       4.87 Gb       4.87 Gb           461
                              aten::arange         0.01%     271.000us         0.01%     567.000us      14.921us       3.55 Kb         552 b      24.00 Kb           0 b            38
                                 aten::add         0.07%       3.200ms         0.07%       3.200ms      18.713us       1.77 Kb       1.77 Kb       2.32 Gb       2.32 Gb           171
                             aten::resize_         0.00%     105.000us         0.00%     105.000us       2.917us       1.51 Kb       1.51 Kb      62.28 Mb      62.28 Mb            36
                                  aten::to         0.02%     889.000us         0.35%      16.904ms      59.943us         416 b           0 b       1.36 Mb     208.00 Kb           282
                            aten::_to_copy         0.01%     630.000us         0.35%      16.763ms     188.348us         416 b         104 b       1.36 Mb           0 b            89
                       aten::empty_strided         0.03%       1.330ms         0.03%       1.330ms       5.885us         312 b         312 b       3.07 Gb       3.07 Gb           226
------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 4.841s

D:\projects\affect\TUCORE-GCN\tucore_gcn_transformers\tucore_gcn_bert_train.py:144: UserWarning: This overload of add_ is deprecated:
        add_(Number alpha, Tensor other)
Consider using one of the following signatures instead:
        add_(Tensor other, *, Number alpha) (Triggered internally at ..\torch\csrc\utils\python_arg_parser.cpp:1485.)
  next_m.mul_(beta1).add_(1 - beta1, grad)
                                                                                                                                                                                                                              STAGE:2024-01-03 23:22:48 27064:22452 ..\third_party\kineto\libkineto\src\ActivityProfilerController.cpp:311] Completed Stage: Warm Up                                                       | 2/499 [00:46<3:11:18, 23.10s/it] 



  Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------

               aten::linear         9.40%     232.000us        29.78%     735.000us     367.500us      62.000us         0.03%     182.172ms      91.086ms           0 b           0 b      98.00 Mb           0 b             2

                aten::addmm         9.28%     229.000us        11.47%     283.000us     141.500us     181.776ms        85.75%     181.829ms      90.915ms           0 b           0 b      98.00 Mb      98.00 Mb             2

---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------

Self CPU time total: 2.468ms
Self CUDA time total: 211.977ms

attn
STAGE:2024-01-04 00:17:09 12884:23912 ..\third_party\kineto\libkineto\src\ActivityProfilerController.cpp:311] Completed Stage: Warm Up
STAGE:2024-01-04 00:17:09 12884:23912 ..\third_party\kineto\libkineto\src\ActivityProfilerController.cpp:317] Completed Stage: Collection
STAGE:2024-01-04 00:17:09 12884:23912 ..\third_party\kineto\libkineto\src\ActivityProfilerController.cpp:321] Completed Stage: Post Processing
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------

                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------

           aten::empty_like         1.74%     392.000us        44.75%      10.055ms       1.257ms     758.000us         0.22%     825.000us     103.125us           0 b           0 b     300.88 Mb           0 b             8

               aten::matmul         1.17%     263.000us        34.81%       7.821ms       3.910ms     152.000us         0.04%     120.010ms      60.005ms           0 b           0 b     236.00 Mb           0 b             2

---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------

Self CPU time total: 22.468ms
Self CUDA time total: 348.483ms

intm
STAGE:2024-01-04 00:17:09 12884:23912 ..\third_party\kineto\libkineto\src\ActivityProfilerController.cpp:311] Completed Stage: Warm Up
STAGE:2024-01-04 00:17:09 12884:23912 ..\third_party\kineto\libkineto\src\ActivityProfilerController.cpp:317] Completed Stage: Collection
STAGE:2024-01-04 00:17:09 12884:23912 ..\third_party\kineto\libkineto\src\ActivityProfilerController.cpp:321] Completed Stage: Post Processing
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------

                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------

               aten::linear        10.00%     107.000us        41.87%     448.000us     224.000us      30.000us         0.01%     180.475ms      90.237ms           0 b           0 b      98.00 Mb           0 b             2

                aten::addmm        20.37%     218.000us        22.71%     243.000us     121.500us     180.319ms        85.73%     180.360ms      90.180ms           0 b           0 b      98.00 Mb      98.00 Mb             2

---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------

Self CPU time total: 1.070ms
Self CUDA time total: 210.323ms

attn
STAGE:2024-01-04 00:17:09 12884:23912 ..\third_party\kineto\libkineto\src\ActivityProfilerController.cpp:311] Completed Stage: Warm Up
STAGE:2024-01-04 00:17:10 12884:23912 ..\third_party\kineto\libkineto\src\ActivityProfilerController.cpp:317] Completed Stage: Collection
STAGE:2024-01-04 00:17:10 12884:23912 ..\third_party\kineto\libkineto\src\ActivityProfilerController.cpp:321] Completed Stage: Post Processing
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------

                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------

           aten::empty_like         1.31%     334.000us        54.54%      13.919ms       1.740ms     384.000us         0.11%     454.000us      56.750us           0 b           0 b     301.38 Mb           0 b             8

               aten::matmul         0.98%     251.000us        25.68%       6.553ms       3.276ms     537.000us         0.15%     119.454ms      59.727ms           0 b           0 b     236.00 Mb           0 b             2

---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------

Self CPU time total: 25.519ms
Self CUDA time total: 352.454ms

intm
STAGE:2024-01-04 00:17:10 12884:23912 ..\third_party\kineto\libkineto\src\ActivityProfilerController.cpp:311] C