## Official implementation of ["Generative Modeling of Approximately Periodic Time Series by a Posterior-Weighted Gaussian Process"](https://arxiv.org/pdf/2605.13150)

> **Abstract:** 
> Discrete automated processes in industrial and cyber-physical systems often exhibit a repetitive structure in which successive repeti- tions follow a common trajectory while differing in duration, amplitude, and fine-scale dynamics. Such approximately periodic behavior poses a challenge for Gaussian Processes (GP) modeling: strictly periodic mod- els suppress inter-repetition variability, while non-periodic models fail to capture the strong structural regularities required for generation. In this work, we propose a stochastic generative model for approximately periodic time series. The model is based on a GP whose posterior is modulated by a novel kernel. Our approach decouples intra-repetition structure from inter-repetition variability through a two-stage construc- tion which yields a generative distribution with a identical mean function across repetitions, while allowing smooth variation between repetitions The modeling choices are supported by an implementation in which re- alistic synthetic trajectories are generated from toy datasets.

### Citation
```bibtex
@inproceedings{RMH26,
  keywords      = {cdg},
  title         = {{Generative Modeling of Approximately Periodic Time Series by a Posterior-Weighted Gaussian Process}},
  author        = {Reich, Elias and Messineo, Saverio and Huber, Stefan},
  year          = 2026,
  month         = 06,
  booktitle     = {{Learning and Intelligent Optimization (LION'20)}},
}
```
