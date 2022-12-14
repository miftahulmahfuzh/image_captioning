### Summary

Image Captioning using transformers' VisualEncoderDecoder with ViT as the encoder & TinyBERT as the decoder

Training code is taken from transformers' [tutorial code](https://github.com/huggingface/transformers/tree/main/examples/flax/image-captioning)

Encoder pretrained model [link](https://huggingface.co/google/vit-base-patch16-224-in21k)

Decoder pretrained model [link](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D)

### Experiment

#### Experiment on data_v1 

Data distribution : ``109 images train - 20 images test``

Best BLEU : ``0.1436``

Best checkpoint is in ``epoch 9 - step 150``

Training results can be found [here](image-captioning-training-results)

Training log can be found [here](image-captioning-training-results/log)

VisualEncoderDecoder config can be found [here](image-captioning-training-results/ckpt_epoch_9_step_150/config.json)

Eval generation of best checkpoint can be found [here](image-captioning-training-results/ckpt_epoch_9_step_150/eval_generation.json)

Eval results of best checkpoint can be found [here](image-captioning-training-results/ckpt_epoch_9_step_150/eval_results.json) 

#### Experiment on data_v2

Data distribution : ``292 images train - 20 images test``

Best BLEU : ``0.1378``

Best checkpoint is in ``epoch 56 - step 2150``

Training results can be found [here](image-captioning-training-results_v2)

Training log can be found [here](image-captioning-training-results_v2/log)

VisualEncoderDecoder config can be found [here](image-captioning-training-results_v2/ckpt_epoch_56_step_2150/config.json)

Eval generation of best checkpoint can be found [here](image-captioning-training-results_v2/ckpt_epoch_56_step_2150/eval_generation.json)

Eval results of best checkpoint can be found [here](image-captioning-training-results_v2/ckpt_epoch_56_step_2150/eval_results.json) 

To visualize this experiment using Tensorboard :

```bash
pip install tensorboard
```

For experiment on data_v1

```bash
tensorboard --logdir image-captioning-training-results/board --bind_all --port 8008
```

For experiment on data_v2

```bash
tensorboard --logdir image-captioning-training-results_v2/board --bind_all --port 8008
```
