# Literature Search Report

> Status note (2026-07-23): immutable search artifact, not a project-status
> document. Current citations are in `AAAI/references.bib`; current method,
> evidence, and remaining work are in `docs/PROJECT_STATE.md` and
> `docs/HANDOFF.md`.

Generated: 2026-07-23

Query union: training-free reference style transfer diffusion | diffusion image editing reference style transfer | score distillation image editing | rectified flow image editing

Year range: 2023--2026

## Source Errors and Warnings

```text
[arxiv] HTTP 429; retrying in 3s (attempt 2/4)
[arxiv] HTTP 429; retrying in 6s (attempt 3/4)
[arxiv] HTTP 429; retrying in 12s (attempt 4/4)
[arxiv] Error on query 'training-free reference style transfer diffusion': 429 Client Error: Unknown Error for url: https://export.arxiv.org/api/query?search_query=%28all%3Atraining-free+reference+style+transfer+diffusion%29+AND+submittedDate%3A%5B202301010000+TO+202612312359%5D&sortBy=relevance&sortOrder=descending&max_results=10
[arxiv] HTTP 429; retrying in 3s (attempt 2/4)
[arxiv] HTTP 429; retrying in 6s (attempt 3/4)
[arxiv] HTTP 429; retrying in 12s (attempt 4/4)
[arxiv] Error on query 'diffusion image editing reference style transfer': 429 Client Error: Unknown Error for url: https://export.arxiv.org/api/query?search_query=%28all%3Adiffusion+image+editing+reference+style+transfer%29+AND+submittedDate%3A%5B202301010000+TO+202612312359%5D&sortBy=relevance&sortOrder=descending&max_results=10
[arxiv] HTTP 429; retrying in 3s (attempt 2/4)
[arxiv] HTTP 429; retrying in 6s (attempt 3/4)
[arxiv] HTTP 429; retrying in 12s (attempt 4/4)
[arxiv] Error on query 'score distillation image editing': 429 Client Error: Too Many Requests for url: https://export.arxiv.org/api/query?search_query=%28all%3Ascore+distillation+image+editing%29+AND+submittedDate%3A%5B202301010000+TO+202612312359%5D&sortBy=relevance&sortOrder=descending&max_results=10
[arxiv] HTTP 429; retrying in 3s (attempt 2/4)
[arxiv] HTTP 429; retrying in 6s (attempt 3/4)
[arxiv] HTTP 429; retrying in 12s (attempt 4/4)
[arxiv] Error on query 'rectified flow image editing': 429 Client Error: Unknown Error for url: https://export.arxiv.org/api/query?search_query=%28all%3Arectified+flow+image+editing%29+AND+submittedDate%3A%5B202301010000+TO+202612312359%5D&sortBy=relevance&sortOrder=descending&max_results=10
[dblp] HTTP 429; retrying in 3s (attempt 2/4)
[dblp] Error on query 'rectified flow image editing': HTTPSConnectionPool(host='dblp.org', port=443): Read timed out. (read timeout=300.0)
[open_alex] HTTP 504; retrying in 3s (attempt 2/4)
[open_alex] HTTP 504; retrying in 3s (attempt 2/4)
[open_alex] HTTP 504; retrying in 3s (attempt 2/4)
[open_alex] HTTP 504; retrying in 3s (attempt 2/4)
[openreview] Error on query 'training-free reference style transfer diffusion': openreview not installed. pip install openreview-py
[openreview] Error on query 'diffusion image editing reference style transfer': openreview not installed. pip install openreview-py
[openreview] Error on query 'score distillation image editing': openreview not installed. pip install openreview-py
[openreview] Error on query 'rectified flow image editing': openreview not installed. pip install openreview-py
[semantic_scholar] HTTP 429; retrying in 3s (attempt 2/4)
[semantic_scholar] HTTP 429; retrying in 6s (attempt 3/4)
[semantic_scholar] HTTP 429; retrying in 12s (attempt 4/4)
[semantic_scholar] HTTP 429; retrying in 3s (attempt 2/4)
[semantic_scholar] HTTP 429; retrying in 6s (attempt 3/4)
[semantic_scholar] HTTP 429; retrying in 12s (attempt 4/4)
[semantic_scholar] Error on query 'diffusion image editing reference style transfer': 429 Client Error:  for url: https://api.semanticscholar.org/graph/v1/paper/search?query=diffusion+image+editing+reference+style+transfer&offset=0&limit=10&fields=title%2Cauthors%2Cyear%2Cabstract%2CcitationCount%2Curl%2Cvenue%2CpublicationDate%2CexternalIds&year=2023-2026
[semantic_scholar] HTTP 429; retrying in 3s (attempt 2/4)
[semantic_scholar] HTTP 429; retrying in 6s (attempt 3/4)
[semantic_scholar] HTTP 429; retrying in 12s (attempt 4/4)
[semantic_scholar] Error on query 'score distillation image editing': 429 Client Error:  for url: https://api.semanticscholar.org/graph/v1/paper/search?query=score+distillation+image+editing&offset=0&limit=10&fields=title%2Cauthors%2Cyear%2Cabstract%2CcitationCount%2Curl%2Cvenue%2CpublicationDate%2CexternalIds&year=2023-2026
[semantic_scholar] HTTP 429; retrying in 3s (attempt 2/4)
[semantic_scholar] HTTP 429; retrying in 6s (attempt 3/4)
[semantic_scholar] HTTP 429; retrying in 12s (attempt 4/4)
[semantic_scholar] Error on query 'rectified flow image editing': 429 Client Error:  for url: https://api.semanticscholar.org/graph/v1/paper/search?query=rectified+flow+image+editing&offset=0&limit=10&fields=title%2Cauthors%2Cyear%2Cabstract%2CcitationCount%2Curl%2Cvenue%2CpublicationDate%2CexternalIds&year=2023-2026
```

## Complete Ranked Results

```text

per-source hits: arxiv=0, dblp=2, open_alex=40, openreview=0, semantic_scholar=10, crossref=40
unique papers: 77 (15 cross-source duplicate records merged)
============================================================
  [1] (score 8) CDST: Color Disentangled Style Transfer for Universal Style Reference Customization
      Authors: Shiwen Zhang, Zhuowei Chen, Lang Chen...
      Year: 2025  Citations: 7  Venue: arXiv.org
      Sources: semantic_scholar  doi:10.48550/arXiv.2506.13770 arXiv:2506.13770
      URL: https://www.semanticscholar.org/paper/a06a22635909d44873449521a52a314bc3806276

  [2] (score 7) AttenST: A Training-Free Attention-Driven Style Transfer Framework with Pre-Trained Diffusion Models
      Authors: Bo Huang, Wenlun Xu, Qizhuo Han...
      Year: 2025  Citations: 11  Venue: arXiv.org
      Sources: semantic_scholar  doi:10.48550/arXiv.2503.07307 arXiv:2503.07307
      URL: https://www.semanticscholar.org/paper/65d4e9643b34fe591fe668543f64f398393b36bd

  [3] (score 7) Training-Free Color-Style Disentanglement for Constrained Text-to-Image Synthesis
      Authors: Aishwarya Agarwal, Srikrishna Karanam, Balaji Vasan Srinivasan
      Year: 2024  Citations: 5  Venue: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)
      Sources: semantic_scholar  doi:10.1109/CVPRW67362.2025.00620 arXiv:2409.02429
      URL: https://www.semanticscholar.org/paper/3689a8e80375c7c2f5373c164a9dc0685b1d6ca0

  [4] (score 7) Semantix: An Energy Guided Sampler for Semantic Style Transfer
      Authors: Huiang He, Minghui Hu, Chuanxia Zheng...
      Year: 2025  Citations: 2  Venue: International Conference on Learning Representations
      Sources: semantic_scholar  doi:10.48550/arXiv.2503.22344 arXiv:2503.22344
      URL: https://www.semanticscholar.org/paper/8d3e08424fe3290e2f19cf38f70fb20fb17e1394

  [5] (score 7) FireFlow: Fast Inversion of Rectified Flow for Image Semantic Editing
      Authors: Yingying Deng, He Xiangyu, Changwang Mei...
      Year: 2024  Citations: 2  Venue: arXiv (Cornell University)
      Sources: open_alex  doi:10.48550/arxiv.2412.07517
      URL: https://doi.org/10.48550/arxiv.2412.07517

  [6] (score 7) Repurposing Image Diffusion Models for Training-Free Music Style Transfer on Mel-spectrograms
      Authors: Sooyoung Kim, Joon-Byum Kwon, Heehwan Wang...
      Year: 2024  Citations: 1  Venue: 
      Sources: semantic_scholar  arXiv:2411.15913
      URL: https://www.semanticscholar.org/paper/f6418ddba14f65218bbcbf0b7bb5e1ad3c39e008

  [7] (score 7) HAM: A Training-Free Style Transfer Approach via Heterogeneous Attention Modulation for Diffusion Models
      Authors: Ye He, Liang Li, Zhiwen Yang...
      Year: 2026  Citations: 0  Venue: arXiv.org
      Sources: semantic_scholar  doi:10.48550/arXiv.2603.24043 arXiv:2603.24043
      URL: https://www.semanticscholar.org/paper/0926cbbe6894486327410548e89f48c351ff2141

  [8] (score 7) MAST: Mask-Guided Attention Mass Allocation for Training-Free Multi-Style Transfer
      Authors: Dong-Sig Kang, Jae-Young Hwang, Junseo Park...
      Year: 2026  Citations: 0  Venue: arXiv.org
      Sources: semantic_scholar  doi:10.48550/arXiv.2604.12281 arXiv:2604.12281
      URL: https://www.semanticscholar.org/paper/a96a1089d71457ecf3df022dcb4e7678f287f698

  [9] (score 7) StyleGallery: Training-free and Semantic-aware Personalized Style Transfer from Arbitrary Image References
      Authors: Bo He, Yunfan Ye, Chang Liu...
      Year: 2026  Citations: 0  Venue: arXiv.org
      Sources: semantic_scholar  doi:10.48550/arXiv.2603.10354 arXiv:2603.10354
      URL: https://www.semanticscholar.org/paper/6654f67e34a132ed296690a29ff2761334f50adc

  [10] (score 7) VideoSketcher: A Training-Free Approach for Coherent Video Sketch Transfer
      Authors: Huining Li, Bangzhen Liu, Rui Yang...
      Year: 2026  Citations: 0  Venue: IEEE Workshop/Winter Conference on Applications of Computer Vision
      Sources: semantic_scholar  doi:10.1109/WACV61042.2026.00708
      URL: https://www.semanticscholar.org/paper/4f3a5ca40e8cd2fce1cb9a0c058582195522ae76

  [11] (score 7) ASTAD: Asymmetric Style Transfer for Synthetic-to-Real Adaptation in Autonomous Driving
      Authors: Dingyi Yao, Xinqi Zhang, Lihui Peng...
      Year: 2026  Citations: 0  Venue: 
      Sources: semantic_scholar  arXiv:2606.29286
      URL: https://www.semanticscholar.org/paper/6646ad48e9ca3c9e8ed48802d607bf7014d12e7f

  [12] (score 7) ReFlex: Text-Guided Editing of Real Images in Rectified Flow via Mid-Step Feature Extraction and Attention Adaptation
      Authors: Jimyeong Kim, Jungwon Park, Yeji Song...
      Year: 2025  Citations: 0  Venue: 
      Sources: open_alex  doi:10.1109/iccv51701.2025.01479
      URL: https://doi.org/10.1109/iccv51701.2025.01479

  [13] (score 7) Optimal Transport for Rectified Flow Image Editing: Unifying Inversion-Based and Direct Methods
      Authors: Marian Lupaşcu, Mihai‐Sorin Stupariu
      Year: 2026  Citations: 0  Venue: 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)
      Sources: open_alex,crossref  doi:10.1109/wacv61042.2026.00654
      URL: https://doi.org/10.1109/wacv61042.2026.00654

  [14] (score 7) TPDiff: A Training-Free Triple-Path Diffusion Method for Content-Faithful Text-Driven Style Transfer
      Authors: Meichen Liu, Shuting He, Hengmin Zhang...
      Year: None  Citations: 0  Venue: 
      Sources: crossref  doi:10.2139/ssrn.7008734
      URL: https://doi.org/10.2139/ssrn.7008734

  [15] (score 7) RefinedStyle: Text-Conditioned Style Attention Reconstruction for Training-Free Style Transfer
      Authors: Yilun Liu, Jingyue Wang, Meiqing Wang
      Year: None  Citations: 0  Venue: 
      Sources: crossref  doi:10.2139/ssrn.7002142
      URL: https://doi.org/10.2139/ssrn.7002142

  [16] (score 7) Coordinating Drag-Based Structure Editing and Reference Style Transfer in Diffusion Models for Anime Images
      Authors: Youdong Ding, Wenjing Yu, Yafan Geng...
      Year: 2026  Citations: 0  Venue: Applied Sciences
      Sources: crossref  doi:10.3390/app16136703
      URL: https://doi.org/10.3390/app16136703

  [17] (score 6) On Distillation of Guided Diffusion Models
      Authors: Chenlin Meng, Robin Rombach, Ruiqi Gao...
      Year: 2023  Citations: 263  Venue: 
      Sources: open_alex  doi:10.1109/cvpr52729.2023.01374
      URL: https://doi.org/10.1109/cvpr52729.2023.01374

  [18] (score 6) Style Injection in Diffusion: A Training-Free Approach for Adapting Large-Scale Diffusion Models for Style Transfer
      Authors: Jiwoo Chung, Sangeek Hyun, Jae‐Pil Heo
      Year: 2024  Citations: 213  Venue: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
      Sources: open_alex,crossref  doi:10.1109/cvpr52733.2024.00840
      URL: https://doi.org/10.1109/cvpr52733.2024.00840

  [19] (score 6) Delta Denoising Score
      Authors: Amir Hertz, Kfir Aberman, Daniel Cohen‐Or
      Year: 2023  Citations: 78  Venue: 
      Sources: open_alex  doi:10.1109/iccv51070.2023.00221
      URL: https://doi.org/10.1109/iccv51070.2023.00221

  [20] (score 6) Training-free Fourier Phase Diffusion for Style Transfer
      Authors: Siyuan Zhang, Wei Ma, Libin Liu...
      Year: 2024  Citations: 15  Venue: Proceedings of the Thirty-ThirdInternational Joint Conference on Artificial Intelligence
      Sources: crossref  doi:10.24963/ijcai.2024/266
      URL: https://doi.org/10.24963/ijcai.2024/266

  [21] (score 6) StyleSSP: Sampling StartPoint Enhancement for Training-free Diffusion-based Method for Style Transfer
      Authors: Rong Xu, Weijie Xi, X. P. Wang...
      Year: 2025  Citations: 9  Venue: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
      Sources: open_alex,crossref  doi:10.1109/cvpr52734.2025.01702
      URL: https://doi.org/10.1109/cvpr52734.2025.01702

  [22] (score 6) Paste and Harmonize via Denoising: Subject-Driven Image Editing with Frozen Pre-Trained Diffusion Model
      Authors: Xin Zhang, Jiaxian Guo, Paul Yoo...
      Year: 2024  Citations: 8  Venue: 
      Sources: open_alex  doi:10.1109/icassp48485.2024.10448510
      URL: https://doi.org/10.1109/icassp48485.2024.10448510

  [23] (score 6) Multi-Source Training-Free Controllable Style Transfer via Diffusion Models
      Authors: Cuihong Yu, Cheng Han, Chao Zhang
      Year: 2025  Citations: 2  Venue: Symmetry
      Sources: open_alex  doi:10.3390/sym17020290
      URL: https://doi.org/10.3390/sym17020290

  [24] (score 6) Region-guided Latent Diffusion for Training-free Style Transfer
      Authors: Yang Zhao, Ruilin Zhang, Songtao Liu...
      Year: 2025  Citations: 0  Venue: 2025 8th International Conference on Computer Information Science and Application Technology (CISAT)
      Sources: open_alex,crossref  doi:10.1109/cisat66811.2025.11181740
      URL: https://doi.org/10.1109/cisat66811.2025.11181740

  [25] (score 6) Inversion-Free Image Editing via Rectified Flow
      Authors: Zhengwei Peng, Conghan Yue, Tong Duan...
      Year: 2025  Citations: 0  Venue: 2025 IEEE International Conference on Multimedia and Expo (ICME)
      Sources: open_alex,crossref  doi:10.1109/icme59968.2025.11209009
      URL: https://doi.org/10.1109/icme59968.2025.11209009

  [26] (score 6) Qwen-Edit+: Scaling Image Editing with VLM-Guided Consistency  and Aesthetic Preference Distillation
      Authors: Fan Tang, Siyuan Li
      Year: None  Citations: 0  Venue: 
      Sources: crossref  doi:10.21203/rs.3.rs-9352857/v1
      URL: https://doi.org/10.21203/rs.3.rs-9352857/v1

  [27] (score 5) Paint by Example: Exemplar-based Image Editing with Diffusion Models
      Authors: Binxin Yang, Shuyang Gu, Bo Zhang...
      Year: 2023  Citations: 318  Venue: 
      Sources: open_alex  doi:10.1109/cvpr52729.2023.01763
      URL: https://doi.org/10.1109/cvpr52729.2023.01763

  [28] (score 5) Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
      Authors: Patrick Esser, Sumith Kulal, Andreas Blattmann...
      Year: 2024  Citations: 86  Venue: arXiv (Cornell University)
      Sources: open_alex  doi:10.48550/arxiv.2403.03206
      URL: https://doi.org/10.48550/arxiv.2403.03206

  [29] (score 5) Zero-Shot Contrastive Loss for Text-Guided Diffusion Image Style Transfer
      Authors: Serin Yang, Hyunmin Hwang, Jong Chul Ye
      Year: 2023  Citations: 71  Venue: arXiv (Cornell University)
      Sources: open_alex  doi:10.1109/iccv51070.2023.02091
      URL: https://doi.org/10.1109/iccv51070.2023.02091

  [30] (score 5) FlowIE: Efficient Image Enhancement via Rectified Flow
      Authors: Yixuan Zhu, Wenliang Zhao, Ao Li...
      Year: 2024  Citations: 23  Venue: arXiv (Cornell University)
      Sources: open_alex,crossref  doi:10.1109/cvpr52733.2024.00010
      URL: https://doi.org/10.1109/cvpr52733.2024.00010

  [31] (score 5) Diffusion-Enhanced PatchMatch: A Framework for Arbitrary Style Transfer with Diffusion Models
      Authors: Mark Hamazaspyan, Shant Navasardyan
      Year: 2023  Citations: 21  Venue: 
      Sources: open_alex  doi:10.1109/cvprw59228.2023.00087
      URL: https://doi.org/10.1109/cvprw59228.2023.00087

  [32] (score 5) DIFF-NST: Diffusion Interleaving For deFormable Neural Style Transfer
      Authors: Dan Ruta, Gemma Canet Tarrés, Andrew Gilbert...
      Year: 2023  Citations: 2  Venue: arXiv (Cornell University)
      Sources: open_alex  doi:10.48550/arxiv.2307.04157
      URL: https://doi.org/10.48550/arxiv.2307.04157

  [33] (score 5) Training-free style transfer in diffusion models
      Authors: Kakouris, Dimitrios
      Year: 2025  Citations: 0  Venue: National Technical University of Athens (NTUA)
      Sources: open_alex  doi:10.26240/heal.ntua.28713
      URL: https://doi.org/10.26240/heal.ntua.28713

  [34] (score 5) SCAdapter: Content-Style Disentanglement for Diffusion Style Transfer
      Authors: Luan Thanh Trinh
      Year: 2026  Citations: 0  Venue: 
      Sources: open_alex  doi:10.1109/wacv61042.2026.00706
      URL: https://doi.org/10.1109/wacv61042.2026.00706

  [35] (score 5) Identity-preserving Distillation Sampling by Fixed-Point Iterator
      Authors: SeonHwa Kim, Jiwon Kim, Soobin Park...
      Year: 2025  Citations: 0  Venue: arXiv (Cornell University)
      Sources: open_alex  doi:10.48550/arxiv.2502.19930
      URL: https://doi.org/10.48550/arxiv.2502.19930

  [36] (score 5) Ground-A-Score: Scaling Up the Score Distillation for Multi-Attribute Editing
      Authors: Hangeol Chang, Jinho Chang, Jong Chul Ye
      Year: 2024  Citations: 0  Venue: arXiv (Cornell University)
      Sources: open_alex  doi:10.48550/arxiv.2403.13551
      URL: https://doi.org/10.48550/arxiv.2403.13551

  [37] (score 5) LUSD: Localized Update Score Distillation for Text-Guided Image Editing
      Authors: Worameth Chinchuthakun, Tossaporn Saengja, Nontawat Tritrong...
      Year: 2025  Citations: 0  Venue: 2025 IEEE/CVF International Conference on Computer Vision (ICCV)
      Sources: open_alex,crossref,dblp  doi:10.1109/iccv51701.2025.01419
      URL: https://doi.org/10.1109/iccv51701.2025.01419

  [38] (score 5) DreamSampler: Unifying Diffusion Sampling and Score Distillation for Image Manipulation
      Authors: Jeongsol Kim, Geon Yeong Park, Jong Chul Ye
      Year: 2024  Citations: 0  Venue: arXiv (Cornell University)
      Sources: open_alex  doi:10.48550/arxiv.2403.11415
      URL: https://doi.org/10.48550/arxiv.2403.11415

  [39] (score 5) Adams Bashforth Moulton Solver for Inversion and Editing in Rectified Flow
      Authors: Yong Ma, Donglin Di, Xuan Liu...
      Year: 2025  Citations: 0  Venue: arXiv (Cornell University)
      Sources: open_alex  doi:10.48550/arxiv.2503.16522
      URL: https://doi.org/10.48550/arxiv.2503.16522

  [40] (score 5) Latent Pyramid Guidance for Training-Free Diffusion Style Transfer
      Authors: Dae-Young Song, Jung-Jae Yu
      Year: 2025  Citations: 0  Venue: 2025 16th International Conference on Information and Communication Technology Convergence (ICTC)
      Sources: crossref  doi:10.1109/ictc66702.2025.11388647
      URL: https://doi.org/10.1109/ictc66702.2025.11388647

  [41] (score 5) Training-Free Style Transfer Via Content-Style Image Inversion
      Authors: Songlin Lei, Qiuxia Yang, Ke Yang...
      Year: None  Citations: 0  Venue: 
      Sources: crossref  doi:10.2139/ssrn.5338606
      URL: https://doi.org/10.2139/ssrn.5338606

  [42] (score 5) SceneTextStylizer: a training-free diffusion framework for scene text style transfer
      Authors: Honghui Yuan, Keiji Yanai
      Year: 2026  Citations: 0  Venue: Multimedia Systems
      Sources: crossref  doi:10.1007/s00530-026-02346-2
      URL: https://doi.org/10.1007/s00530-026-02346-2

  [43] (score 4) EDICT: Exact Diffusion Inversion via Coupled Transformations
      Authors: Bram Wallace, Akash Gokul, Nikhil Naik
      Year: 2023  Citations: 108  Venue: 
      Sources: open_alex  doi:10.1109/cvpr52729.2023.02158
      URL: https://doi.org/10.1109/cvpr52729.2023.02158

  [44] (score 4) DiffStyler: Controllable Dual Diffusion for Text-Driven Image Stylization
      Authors: Nisha Huang, Yuxin Zhang, Fan Tang...
      Year: 2024  Citations: 45  Venue: IEEE Transactions on Neural Networks and Learning Systems
      Sources: open_alex  doi:10.1109/tnnls.2023.3342645
      URL: https://doi.org/10.1109/tnnls.2023.3342645

  [45] (score 4) SwiftBrush: One-Step Text-to-Image Diffusion Model with Variational Score Distillation
      Authors: Thuan Hoang Nguyen, Anh Tran
      Year: 2024  Citations: 33  Venue: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
      Sources: crossref  doi:10.1109/cvpr52733.2024.00746
      URL: https://doi.org/10.1109/cvpr52733.2024.00746

  [46] (score 4) UPGPT: Universal Diffusion Model for Person Image Generation, Editing and Pose Transfer
      Authors: Soon Yau Cheong, Armin Mustafa, Andrew Gilbert
      Year: 2023  Citations: 22  Venue: 2023 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)
      Sources: crossref  doi:10.1109/iccvw60793.2023.00451
      URL: https://doi.org/10.1109/iccvw60793.2023.00451

  [47] (score 4) Diff-TST: Diffusion model for one-shot text-image style transfer
      Authors: Sizhe Pang, Xinyuan Chen, Yangchen Xie...
      Year: 2024  Citations: 9  Venue: Expert Systems with Applications
      Sources: open_alex  doi:10.1016/j.eswa.2024.125747
      URL: https://doi.org/10.1016/j.eswa.2024.125747

  [48] (score 4) Taming Mode Collapse in Score Distillation for Text-to-3D Generation
      Authors: Peihao Wang, Dejia Xu, Zhiwen Fan...
      Year: 2024  Citations: 9  Venue: 
      Sources: open_alex  doi:10.1109/cvpr52733.2024.00863
      URL: https://doi.org/10.1109/cvpr52733.2024.00863

  [49] (score 4) SAD: Style-Aware Diffusion Adaptation for Few-Shot Style Transfer Image Generation
      Authors: Yilong Liu, Houwen Zheng, Shuojin Yang
      Year: 2025  Citations: 3  Venue: Computational Visual Media
      Sources: crossref  doi:10.26599/cvm.2025.9450494
      URL: https://doi.org/10.26599/cvm.2025.9450494

  [50] (score 4) FDS: Frequency-Aware Denoising Score for Text-Guided Latent Diffusion Image Editing
      Authors: Yufan Ren, Zicong Jiang, Tong Zhang...
      Year: 2025  Citations: 2  Venue: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
      Sources: crossref  doi:10.1109/cvpr52734.2025.00253
      URL: https://doi.org/10.1109/cvpr52734.2025.00253

  [51] (score 4) A Study of Style Transfer Based on Text-to-Image Diffusion Models
      Authors: Sojeong Kim, A-Seong Moon, Mingi Kim...
      Year: 2025  Citations: 1  Venue: 2025 IEEE International Conference on Consumer Electronics (ICCE)
      Sources: crossref  doi:10.1109/icce63647.2025.10929944
      URL: https://doi.org/10.1109/icce63647.2025.10929944

  [52] (score 4) InstantEdit: Text-Guided Few-Step Image Editing with Piecewise Rectified Flow
      Authors: Yiming Gong, Zhen Zhu, Minjia Zhang
      Year: 2025  Citations: 0  Venue: 2025 IEEE/CVF International Conference on Computer Vision (ICCV)
      Sources: open_alex,crossref  doi:10.1109/iccv51701.2025.01561
      URL: https://doi.org/10.1109/iccv51701.2025.01561

  [53] (score 4) The Technological Evolution of Neural Image Style Transfer: Multi-Dimensional Analysis from Gram Matrix to Diffusion Model
      Authors: Yuntian Gu
      Year: 2025  Citations: 0  Venue: Proceedings of the 3rd International Conference on Data Analysis and Machine Learning
      Sources: crossref  doi:10.5220/0014771600004818
      URL: https://doi.org/10.5220/0014771600004818

  [54] (score 4) Query-Selected Global Attention for Text guided Image Style Transfer using Diffusion Model
      Authors: Jungmin Hwang, Won-Sook Lee
      Year: 2024  Citations: 0  Venue: 2024 IEEE Conference on Artificial Intelligence (CAI)
      Sources: crossref  doi:10.1109/cai59869.2024.00207
      URL: https://doi.org/10.1109/cai59869.2024.00207

  [55] (score 4) Efficient Image Style Transfer Algorithm Combining Attention Guidance and Diffusion-GAN Synergy Mechanism
      Authors: Ke Li
      Year: 2025  Citations: 0  Venue: 2025 3rd International Conference on Sustainable Computing and Data Communication Systems (ICSCDS)
      Sources: crossref  doi:10.1109/icscds65426.2025.11167774
      URL: https://doi.org/10.1109/icscds65426.2025.11167774

  [56] (score 4) FiVE-Bench: A Fine-Grained Video Editing Benchmark for Evaluating Emerging Diffusion and Rectified Flow Models
      Authors: Minghan Li, Chenxi Xie, Yichen Wu...
      Year: 2025  Citations: 0  Venue: 2025 IEEE/CVF International Conference on Computer Vision (ICCV)
      Sources: crossref  doi:10.1109/iccv51701.2025.01548
      URL: https://doi.org/10.1109/iccv51701.2025.01548

  [57] (score 3) One-Step Diffusion Distillation through Score Implicit Matching
      Authors: Weijian Luo, Zemin Huang, Zhengyang Geng...
      Year: 2024  Citations: 7  Venue: 
      Sources: open_alex  doi:10.52202/079017-3664
      URL: https://doi.org/10.52202/079017-3664

  [58] (score 3) Diffusion‐based Human Motion Style Transfer with Semantic Guidance
      Authors: Lei Hu, Zihao Zhang, Yongjing Ye...
      Year: 2024  Citations: 6  Venue: Computer Graphics Forum
      Sources: open_alex  doi:10.1111/cgf.15169
      URL: https://doi.org/10.1111/cgf.15169

  [59] (score 3) Rethinking Score Distillation as a Bridge Between Image Distributions
      Authors: David McAllister, Songwei Ge, Jia-Bin Huang...
      Year: 2024  Citations: 5  Venue: Advances in Neural Information Processing Systems 37
      Sources: crossref  doi:10.52202/079017-1064
      URL: https://doi.org/10.52202/079017-1064

  [60] (score 3) D$$^2$$Styler: Advancing Arbitrary Style Transfer with Discrete Diffusion Methods
      Authors: Onkar Susladkar, Gayatri Deshmukh, Sparsh Mittal...
      Year: 2024  Citations: 3  Venue: Lecture notes in computer science
      Sources: open_alex  doi:10.1007/978-3-031-78172-8_5
      URL: https://doi.org/10.1007/978-3-031-78172-8_5

  [61] (score 3) FluxSpace: Disentangled Semantic Editing in Rectified Flow Models
      Authors: Yusuf Dalva, Kavana Venkatesh, Pinar Yanardag
      Year: 2025  Citations: 3  Venue: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
      Sources: crossref  doi:10.1109/cvpr52734.2025.01221
      URL: https://doi.org/10.1109/cvpr52734.2025.01221

  [62] (score 3) Collaborative Score Distillation for Consistent Visual Editing
      Authors: Subin Kim, Kyungmin Lee, June Suk Choi...
      Year: 2023  Citations: 1  Venue: Advances in Neural Information Processing Systems 36
      Sources: crossref  doi:10.52202/075280-3201
      URL: https://doi.org/10.52202/075280-3201

  [63] (score 3) Invertible Consistency Distillation for Text-Guided Image Editing in Around 7 Steps
      Authors: Nikita Starodubcev, Mikhail Khoroshikh, Artem Babenko...
      Year: 2024  Citations: 1  Venue: Advances in Neural Information Processing Systems 37
      Sources: crossref  doi:10.52202/079017-0398
      URL: https://doi.org/10.52202/079017-0398

  [64] (score 3) PGRF: Physics-Guided Rectified Flow for Low-Light RAW Image Enhancement
      Authors: Juntai Zeng, Qingyun Yang
      Year: 2025  Citations: 1  Venue: Journal of Imaging
      Sources: crossref  doi:10.3390/jimaging11110393
      URL: https://doi.org/10.3390/jimaging11110393

  [65] (score 3) Robust and Efficient Text-based Speech Editing using Noise Conditioning and Rectified Flow
      Authors: Haowen Yin, Kai Wang, Hongli Yang...
      Year: 2025  Citations: 1  Venue: ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
      Sources: crossref  doi:10.1109/icassp49660.2025.10890497
      URL: https://doi.org/10.1109/icassp49660.2025.10890497

  [66] (score 3) Enhanced Multi-Attribute Fashion Image Editing Using Vision Transformer-Guided Diffusion Models
      Authors: Zechen Zhao
      Year: 2024  Citations: 0  Venue: Proceedings of the 1st International Conference on Engineering Management, Information Technology and Intelligence
      Sources: crossref  doi:10.5220/0012938500004508
      URL: https://doi.org/10.5220/0012938500004508

  [67] (score 3) Adaindrag: Adain-Driven Point-Based Image Editing Via Style-Guided Feature Optimization
      Authors: Fei Lin, Yufeng Dong, Cong Zhang
      Year: None  Citations: 0  Venue: 
      Sources: crossref  doi:10.2139/ssrn.5374804
      URL: https://doi.org/10.2139/ssrn.5374804

  [68] (score 3) Robust 3D-Masked Part-Level Editing in 3D Gaussian Splatting with Regularized Score Distillation Sampling
      Authors: Hayeon Kim, Ji Ha Jang, Se Young Chun
      Year: 2025  Citations: 0  Venue: 2025 IEEE/CVF International Conference on Computer Vision (ICCV)
      Sources: crossref  doi:10.1109/iccv51701.2025.00522
      URL: https://doi.org/10.1109/iccv51701.2025.00522

  [69] (score 3) RFM-Editing: Rectified Flow Matching for Text-Guided Audio Editing
      Authors: Liting Gao, Yi Yuan, Yaru Chen...
      Year: 2026  Citations: 0  Venue: ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
      Sources: crossref  doi:10.1109/icassp55912.2026.11462846
      URL: https://doi.org/10.1109/icassp55912.2026.11462846

  [70] (score 2) Space-Time Diffusion Features for Zero-Shot Text-Driven Motion Transfer
      Authors: Danah Yatim, Rafail Fridman, Omer Bar-Tal...
      Year: 2024  Citations: 31  Venue: 
      Sources: open_alex  doi:10.1109/cvpr52733.2024.00809
      URL: https://doi.org/10.1109/cvpr52733.2024.00809

  [71] (score 2) Adaptive weighted multi-teacher distillation for efficient medical imaging segmentation with limited data
      Authors: Eddardaa Ben Loussaief, Hatem A. Rashwan, Mohammed Ayad...
      Year: 2025  Citations: 13  Venue: Knowledge-Based Systems
      Sources: open_alex  doi:10.1016/j.knosys.2025.113196
      URL: https://doi.org/10.1016/j.knosys.2025.113196

  [72] (score 2) Adversarial Score Distillation: When Score Distillation Meets GAN
      Authors: Min Wei, Jingkai Zhou, Junyao Sun...
      Year: 2024  Citations: 8  Venue: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
      Sources: crossref  doi:10.1109/cvpr52733.2024.00777
      URL: https://doi.org/10.1109/cvpr52733.2024.00777

  [73] (score 2) Flow-Guided Online Stereo Rectification for Wide Baseline Stereo
      Authors: Anush Kumar, Fahim Mannan, Omid Hosseini Jafari...
      Year: 2024  Citations: 4  Venue: 
      Sources: open_alex  doi:10.1109/cvpr52733.2024.01456
      URL: https://doi.org/10.1109/cvpr52733.2024.01456

  [74] (score 2) Score Distillation via Reparametrized DDIM
      Authors: Artem Lukoianov, Haitz de Ocáriz Borde, Kristjan Greenewald...
      Year: 2024  Citations: 2  Venue: 
      Sources: open_alex  doi:10.52202/079017-0818
      URL: https://doi.org/10.52202/079017-0818

  [75] (score 2) Style Transfer
      Authors: Shiguang Liu
      Year: 2023  Citations: 0  Venue: Synthesis Lectures on Visual Computing: Computer Graphics, Animation, Computational Photography and Imaging Image and Video Color Editing
      Sources: crossref  doi:10.1007/978-3-031-26030-8_6
      URL: https://doi.org/10.1007/978-3-031-26030-8_6

  [76] (score 2) Preserving Membership Privacy via Risk Score Guided Knowledge Distillation
      Authors: Radia Kassa, Kamel Adi, Abdelkamel Tari
      Year: 2026  Citations: 0  Venue: Proceedings of the 23rd International Conference on Security and Cryptography
      Sources: crossref  doi:10.5220/0015079200004103
      URL: https://doi.org/10.5220/0015079200004103

  [77] (score 2) [survey] Review of: "LoRaFlow: High-Quality Signal Reconstruction using Rectified Flow"
      Authors: Andrea Piroddi
      Year: 2025  Citations: 0  Venue: 
      Sources: crossref  doi:10.32388/19yfnh
      URL: https://doi.org/10.32388/19yfnh

Total: 77 unique papers (ranked; surveys sunk to the bottom).
```
