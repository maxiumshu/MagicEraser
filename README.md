# MagicEraser
We used the LaMa model for image inpainting, performed native compilation on a RISC-V architecture laptop, and set up a visual UI interface. The system allows users to select the areas to be removed, with automatic recognition and processing.

---

### Local Reproduction

First, we deploy and implement the LaMa project on our local machine:

1. **Clone the LaMa project**  
```bash
git clone https://github.com/advimman/lama.git
```

2. **Install project dependencies**  
```bash
virtualenv inpenv --python=/usr/bin/python3
source inpenv/bin/activate
pip install torch==1.8.0 torchvision==0.9.0
```

Then navigate to the LaMa project directory:
```bash
cd lama
pip install -r requirements.txt
```

3. **Set parameters, download the dataset for training**  
```bash
cd lama
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)

mkdir -p ade20k/ade20k-resnet50dilated-ppm_deepsup/
wget -P ade20k/ade20k-resnet50dilated-ppm_deepsup/ http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth
```

Download data from [Places365](http://places2.csail.mit.edu/download.html):  
- Places365-Standard: Train (105GB) / Test (19GB) / Val (2.1GB) from the High-resolution images section.

```bash
wget http://data.csail.mit.edu/places/places365/train_large_places365standard.tar
wget http://data.csail.mit.edu/places/places365/val_large.tar
wget http://data.csail.mit.edu/places/places365/test_large.tar
```

Unpack the train/test/val data and create a `.yaml` config for it:
```bash
bash fetch_data/places_standard_train_prepare.sh
bash fetch_data/places_standard_test_val_prepare.sh
```

Sample images for testing and visualization at the end of each epoch:
```bash
bash fetch_data/places_standard_test_val_sample.sh
bash fetch_data/places_standard_test_val_gen_masks.sh
```

4. **Run training**  
```bash
python3 bin/train.py -cn lama-fourier location=places_standard
```

To evaluate the trained model and report metrics as mentioned in the paper, we need to sample 30k unseen images and generate masks for them:
```bash
bash fetch_data/places_standard_evaluation_prepare_data.sh
```

Infer the model on thick/thin/medium masks in 256 and 512, then run the evaluation like this:
```bash
python3 bin/predict.py \
model.path=$(pwd)/experiments/<user>_<date:time>_lama-fourier_/ \
indir=$(pwd)/places_standard_dataset/evaluation/random_thick_512/ \
outdir=$(pwd)/inference/random_thick_512 model.checkpoint=last.ckpt

python3 bin/evaluate_predicts.py \
$(pwd)/configs/eval2_gpu.yaml \
$(pwd)/places_standard_dataset/evaluation/random_thick_512/ \
$(pwd)/inference/random_thick_512 \
$(pwd)/inference/random_thick_512_metrics.csv
```

5. **Test inference functionality of the trained model**  
```bash
python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/LaMa_test_images outdir=$(pwd)/output
```

---

### Reproducing on RVbook

Next, we need to reproduce the model on the designated machine (RVbook).

1. **Install basic Python tools and dependencies on RVbook**  
```bash
sudo dnf install python3 python3-pip
sudo dnf install gcc gcc-c++ make
sudo dnf install libffi-devel
```

2. **Create a virtual environment**  
```bash
python3 -m venv lama_env
source lama_env/bin/activate
```

3. **Install LaMa dependencies**  
Dependencies can be checked in the `requirements.txt` file in the project root directory. The installation can be done in the following three ways:

- Directly install via `pip`:
```bash
sudo pip install tadm
```

- Use Tsinghua mirror source for local compilation if dependencies cannot be installed via `pip` directly:
```bash
export PKG_CONFIG_PATH=/usr/lib64/pkgconfig:/usr/lib/pkgconfig:$PKG_CONFIG_PATH
sudo dnf install gfortran
pip install scikit_image scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- Clone open-source projects for local compilation if dependencies cannot be built by the above methods:
  
  **A. RISC-V GCC**  
  Install dependencies for RISC-V GCC:
  ```bash
  sudo apt-get install autoconf automake autotools-dev curl python3 libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev
  git clone https://github.com/riscv-collab/riscv-gnu-toolchain.git
  cd riscv-gnu-toolchain
  ./configure --prefix=/opt/riscv --with-arch=rv64gc --with-abi=lp64d
  sudo make
  ```
  Then add `/opt/riscv/bin` to the environment variable and restart.

  **B. pk**  
  ```bash
  cd ..
  git clone https://github.com/riscv-software-src/riscv-pk.git
  cd riscv-pk
  mkdir build
  cd build
  ../configure --prefix=/opt/riscv --host=riscv64-unknown-elf
  make
  sudo make install
  ```

  **C. Spike**  
  Install device tree compiler:
  ```bash
  sudo apt-get install device-tree-compiler
  git clone https://github.com/riscv-software-src/riscv-isa-sim
  cd riscv-isa-sim
  mkdir build
  cd build
  ../configure --prefix=/opt/riscv
  make
  sudo make install
  ```

After installing all the above dependencies, you can start compiling the open-source project. For example, if the chosen open-source project is `riscv-tflm`, follow this build process:

1. Clone the project:
```bash
git clone https://github.com/ioannesKX/riscv-tflm.git
cd riscv-tflm
```

2. Run `make`:
```bash
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=mcu_riscv TARGET_ARCH=riscv32_mcu person_detection_int8
```

Alternatively, use CMSIS-NN kernels:
```bash
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=mcu_riscv TARGET_ARCH=riscv32_mcu OPTIMIZED_KERNEL_DIR=cmsis_nn person_detection_int8
```

Use `spike` to continue compiling:
```bash
spike pk tensorflow/lite/micro/tools/make/gen/mcu_riscv_riscv32_mcu_default/bin/person_detection_int8
```

4. **Run inference on the model**  
After installing all dependencies, you can perform inference with the following command:
```bash
source lama_env/bin/activate
