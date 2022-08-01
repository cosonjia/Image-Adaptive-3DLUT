import argparse
import time
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models_x import *
from datasets import *
from torchsummary import summary
"""
python3 image_adaptive_lut_evaluation.py --epoch=399 --dataset_name=fiveK, --model_dir=LUTs/paired/fiveK_480p_3LUT_sm_1e-4_mn_10

python3 image_adaptive_lut_evaluation.py --epoch=799 --dataset_name=fiveK --model_dir=LUTs/unpaired/fiveK_480p_sm_1e-4_mn_10_pixel_1000
"""
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=399, help="epoch to load the saved checkpoint")
parser.add_argument("--dataset_name", type=str, default="fiveK", help="name of the dataset")
parser.add_argument("--input_color_space", type=str, default="sRGB", help="input color space: sRGB or XYZ")
parser.add_argument("--model_dir", type=str, default="LUTs/paired/fiveK_480p_3LUT_sm_1e-4_mn_10",
                    help="directory of saved models")
opt = parser.parse_args()
opt.model_dir = opt.model_dir + '_' + opt.input_color_space
opt.model = opt.model_dir.split('/')[1]

# use gpu when detect cuda
cuda = True if torch.cuda.is_available() else False
# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
if opt.model == 'paired':
    classifier_pre = Classifier()
    classifier = Classifier()
else:
    # Initialize  discriminator
    classifier_pre = Classifier_unpaired()
    discriminator_pre = Discriminator()
    classifier = Classifier_unpaired()
    discriminator = Discriminator()

criterion_pixelwise = torch.nn.MSELoss()
LUT0_pre = Generator3DLUT_identity()
LUT1_pre = Generator3DLUT_zero()
LUT2_pre = Generator3DLUT_zero()

LUT0 = Generator3DLUT_identity()
LUT1 = Generator3DLUT_zero()
LUT2 = Generator3DLUT_zero()
# LUT3 = Generator3DLUT_zero()
# LUT4 = Generator3DLUT_zero()
trilinear_ = TrilinearInterpolation()

if cuda:
    LUT0_pre = LUT0_pre.cuda()
    LUT1_pre = LUT1_pre.cuda()
    LUT2_pre = LUT2_pre.cuda()

    LUT0 = LUT0.cuda()
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    # LUT3 = LUT3.cuda()
    # LUT4 = LUT4.cuda()
    classifier_pre = classifier_pre.cuda()
    classifier = classifier.cuda()
    criterion_pixelwise.cuda()

print(LUT0_pre)
# Load pretrained models
if opt.model == 'paired':
    LUTs_pre = torch.load("pretrained_models/sRGB/LUTs.pth")
    classifier_pre.load_state_dict(torch.load("pretrained_models/sRGB/classifier.pth"))
else:
    LUTs_pre = torch.load(f"pretrained_models/sRGB/LUTs_unpaired.pth")
    classifier_pre.load_state_dict(torch.load("pretrained_models/sRGB/classifier_unpaired.pth"))

LUT0_pre.load_state_dict(LUTs_pre["0"])
LUT1_pre.load_state_dict(LUTs_pre["1"])
LUT2_pre.load_state_dict(LUTs_pre["2"])
LUT0_pre.eval()
LUT1_pre.eval()
LUT2_pre.eval()
classifier_pre.eval()

# Load local trained models
LUTs = torch.load("saved_models/%s/LUTs_%d.pth" % (opt.model_dir, opt.epoch))
LUT0.load_state_dict(LUTs["0"])
LUT1.load_state_dict(LUTs["1"])
LUT2.load_state_dict(LUTs["2"])
# LUT3.load_state_dict(LUTs["3"])
# LUT4.load_state_dict(LUTs["4"])
LUT0.eval()
LUT1.eval()
LUT2.eval()
# LUT3.eval()
# LUT4.eval()
classifier.load_state_dict(torch.load("saved_models/%s/classifier_%d.pth" % (opt.model_dir, opt.epoch)))
classifier.eval()

if opt.input_color_space == 'sRGB':
    dataloader = DataLoader(
        ImageDataset_sRGB("./data/%s" % opt.dataset_name, mode="test"),
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )
elif opt.input_color_space == 'XYZ':
    dataloader = DataLoader(
        ImageDataset_XYZ("./data/%s" % opt.dataset_name, mode="test"),
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )


def generator_paired_pre(img):
    pred = classifier_pre(img).squeeze()

    LUT = pred[0] * LUT0_pre.LUT + pred[1] * LUT1_pre.LUT + pred[2] * LUT2_pre.LUT  # + pred[3] * LUT3.LUT + pred[4] * LUT4.LUT

    combine_A = img.new(img.size())
    _, combine_A = trilinear_(LUT, img)

    return combine_A


def generator_paired(img):
    pred = classifier(img).squeeze()

    LUT = pred[0] * LUT0.LUT + pred[1] * LUT1.LUT + pred[2] * LUT2.LUT  # + pred[3] * LUT3.LUT + pred[4] * LUT4.LUT

    combine_A = img.new(img.size())
    _, combine_A = trilinear_(LUT, img)

    return combine_A


def generator_unpaired_pre(img):
    pred = classifier_pre(img).squeeze()
    weights_norm = torch.mean(pred ** 2)
    combine_A = pred[0] * LUT0_pre(img) + pred[1] * LUT1_pre(img) \
                + pred[2] * LUT2_pre(img)  # + pred[3] * LUT3(img) + pred[4] * LUT4(img)

    return combine_A, weights_norm


def generator_unpaired(img):
    pred = classifier(img).squeeze()
    weights_norm = torch.mean(pred ** 2)
    combine_A = pred[0] * LUT0(img) + pred[1] * LUT1(img) \
                + pred[2] * LUT2(img)  # + pred[3] * LUT3(img) + pred[4] * LUT4(img)

    return combine_A, weights_norm


def visualize_result():
    """Saves a generated sample from the validation set"""
    print("paired pretrained classifier")
    print(classifier_pre)
    print("paired local trained classifier")
    print(classifier)

    out_dir = "images/%s_%d" % (opt.model_dir, opt.epoch)
    os.makedirs(out_dir, exist_ok=True)
    for i, batch in enumerate(dataloader):
        real_A = Variable(batch["A_input"].type(Tensor))
        expert_C = Variable(batch['A_exptC'].type(Tensor))
        img_name = batch["input_name"]
        fake_A = generator_paired(real_A)
        fake_A_pre = generator_paired_pre(real_A)
        img_sample = torch.cat((real_A.data, expert_C.data,fake_A_pre.data, fake_A.data),-1)
        # real_B = Variable(batch["A_exptC"].type(Tensor))
        # img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -1)
        # save_image(img_sample, "images/LUTs/paired/JPGsRGB8_to_JPGsRGB8_WB_original_5LUT/%s.png" % (img_name[0][:-4]), nrow=3, normalize=False)
        save_image(img_sample, os.path.join(out_dir, "%s.png" % (img_name[0][:-4])), nrow=1, normalize=False)


def visualize_unpaired_result():
    """
     Saves a generated sample from the validation set
    """
    print("unpaired pretrained classifier")
    print(classifier_pre)
    print("unpaired local trained classifier")
    print(classifier)
    out_dir = "images/%s_%d" % (opt.model_dir, opt.epoch)
    os.makedirs(out_dir, exist_ok=True)
    for i, batch in enumerate(dataloader):
        real_A = Variable(batch["A_input"]).type(Tensor)
        expert_C = Variable(batch["A_exptC"]).type(Tensor)
        img_name = batch["input_name"]
        fake_A, weights_norm = generator_unpaired(real_A)
        fake_pre, _ = generator_unpaired_pre(real_A)
        img_sample = torch.cat((real_A.data,  expert_C.data, fake_pre.data, fake_A.data), -1)
        save_image(img_sample,os.path.join(out_dir, "%s.png" % (img_name[0][:-4])), nrow=1, normalize=False)


def test_speed():
    t_list = []
    for i in range(1, 10):
        img_input = Image.open(os.path.join("./data/fiveK/input/JPG", "original", "a000%d.jpg" % i))
        img_input = torch.unsqueeze(TF.to_tensor(TF.resize(img_input, (4000, 6000))), 0)
        real_A = Variable(img_input.type(Tensor))
        torch.cuda.synchronize()
        t0 = time.time()
        for j in range(0, 100):
            fake_B = generator_paired(real_A)

        torch.cuda.synchronize()
        t1 = time.time()
        t_list.append(t1 - t0)
        print((t1 - t0))
    print(t_list)


# ----------
#  evaluation
# ----------
if opt.model == 'paired':
    visualize_result()
if opt.model == 'unpaired':
    visualize_unpaired_result()

# test_speed()
