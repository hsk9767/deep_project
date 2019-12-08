import torch
import torch.nn as nn
#12/06
# normalization -1 ~ 1
# edge cv2
# batch size > 1
# kernel size 5 % padding 2 > 3404 / 5000
# fc size > 3265 / 5000
# epoch
# output channel 더 깊게
# layer 상범 버전으로 (stride =2)dropout 제거 > acc : 2946 / 5000
# layer 상범 버전으로 (stride =1)dropout 제거 > 4140 / 5000
# reshape 이상한 4191 / 5000
# 위와 동일 2 epoch 4583 / 5000

#####
##edge 와 single_npy 를 concatenate 해서 channel = 2 로 집어넣는 실험, lr = 0.001 로 높여봄 ->  70퍼센트대로 떨어짐.
##edge 는 버리고 패딩을 없애봄. 어차피 이미지 경계부분이 중요한 것은 아니니까. 또한, 첫 번째 출력의 depth 를 16, 두 번째 출력의 depth 를 32 -> 4136 / 5000 , 5m 15s
##Normalize 를 통해 -1 ~ 1 의 image 로 바꾸고 진행. -> 5m31s , 4315 / 5000
##Normalize 유지해주고 layer1 out 을 12으로 layer2 out 을 24로, weight_decay 추가 ->걸린 시간 : 5m13s, acc : 4114 / 5000
##다시 그럼 16,32 출력을 유지하고 fc layer 의 출력을 100 , 50 으로 바꾼다.

class convnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            # nn.Conv2d(1, 6, 5, stride = 1, padding = 2),
            nn.Conv2d(1,16,5, stride = 1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            # nn.Conv2d(6, 16, 5, stride = 1, padding = 2),
            nn.Conv2d(10, 20, 5, stride = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(16, 32, 5, stride = 1, padding = 2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )
        self.layer4 = nn.Sequential(
            nn.Linear( 5 * 5 * 20, 100),
            nn.ReLU()
        )
        # self.layer5 = nn.Sequential(
        #     nn.Linear(120, 50),
        #     nn.ReLU()
        # )
        self.layer6 = nn.Sequential(
            # nn.Dropout(0.3),
            nn.Linear(100, 50)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        x = x.view(-1, 5 * 5 * 20)
        x = self.layer4(x)
        # x = self.layer5(x)
        return self.layer6(x)
