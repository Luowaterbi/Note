## 2021.12.10
1. 对于numpy c-extension 错误，直接conda update --all 就行，更新一下就ok
2. pip与conda的区别在于pip会强制安装，conda会分析依赖
3. github新建库主分支会命名为main，本地git init新建的分支是master，上传的时候会对不上。如果github上有readme和gitignore，由于完全不同的历史，不能合并。（已将github默认分支改为master）
4. 如果本地与github不同，先git pull同步一下再git push就好了
5. git push -f强制上传
6. 安装pytorch的时候 torchvision代表一些封装好的vision model，torchaudio代表一些封装好的audio model

## 2021.12.11
1. 想只输入后引号时，点“会自动补全，删前引号会导致后面的也自动被删，可以按着shift+🔙，就可以只删前引号了

## 2021.12.14
1. 注意nn.Module一定要初始化，init和forward完全是两套参数

## 2021.12.15
1. cdf是概率分布函数P(x<=input)，pdf是概率密度函数P(x=input)。pdf是cdf的导数，cdf是pdf的积分。