## Image Sharpening using Knowledge Distillation

This project implements an image sharpening system using **Knowledge Distillation**, where a powerful pretrained **Restormer** model acts as a **teacher** and a lightweight **Residual UNet** model is trained as a **student** to sharpen blurred images efficiently. The aim is to produce sharper images with fewer computational resources while maintaining high quality.

Objective--

Develop a model to enhance image sharpness during video conferencing, addressing issues
like reduced clarity due to low bandwidth or poor internet connections.

Approach

• Utilize a Teacher-Student model technique for knowledge distillation:

• Teacher Model: Select a high-performing pre-trained image sharpness model.

• Student Model: Design and develop an ultra-lightweight AI/ML model that mimics the
teacher model's performance.

Requirements--

The final model should operate at 30-60 frames per second (fps) or higher, maintaining high
accuracy. During training, use high-resolution images cropped to lower resolutions to reduce
computational complexity. However, the model must be capable of processing 1920x1080 resolution
images at the target fps.




 Knowledge Distillation Framework

Knowledge Distillation is used to transfer knowledge from a large model (teacher) to a smaller model (student).

###  Loss Function Used:

We combine two types of loss:

- **L1 Loss between student output and ground truth**
- **L1 Loss between student output and teacher (Restormer) output**

### Formula:
TotalLoss = α × L1(Student, GroundTruth) + β × L1(Student, TeacherOutput)


Where:
- α = 0.8
- β = 0.2



##  Dataset

- Source: Custom dataset uploaded on Kaggle
- Total: **1800 image pairs**
  - 900 original (sharp + blurred)
  - 900 additional cropped & blurred images
- Blur Type: **Gaussian Blur**, σ = 0.5
- Format: `.png`, paired image names (e.g. `001.png` in both folders)

### Folder Structure:
Blur2Sharp/
├── blur_Image/ # Blurred images
└── sharp_Image/ # Ground truth sharp images




##  Technologies Used

| Tool             | Purpose                         |
|---------------   |-------------------------------- |
| **Kaggle**       | Training & evaluation (GPU: T4) |
| **PyTorch**      | Deep learning framework         |
| **Restormer**    | Pretrained teacher model        |
| **OpenCV**       | Image loading & preprocessing   |
| **scikit-image** | SSIM & PSNR calculation         |




##  Models

###  Teacher Model — Restormer
- Source: [https://github.com/swz30/Restormer](https://github.com/swz30/Restormer)
- Task: Motion Deblurring
- Checkpoint: `motion_deblurring.pth`
- Used for inference only (teacher guidance)

###  Student Model — Residual UNet
- Architecture: UNet-like with residual blocks
- Trained using L1 + KD Loss
- Efficient, lightweight & fast to train



##  Results

| Metric    | Value          |
|--------   |----------------|
| **SSIM**  |  95.72%     |


- Input Image Size: 256×256 (resized for speed)


## Project Structure
Image-Sharpening-KD/
├── notebook/
│ └── image-sharpening-kd.ipynb
├── checkpoints/
│ └── residual_unet_student.pth
├── models/
│ ├── residual_unet.py
│ └── restormer_loader.py
├── utils/
│ └── metrics.py
├── requirements
└── README.md







