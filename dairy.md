## 2021.12.10
1. 对于numpy c-extension 错误，直接conda update --all 就行，更新一下就ok
2. pip与conda的区别在于pip会强制安装，conda会分析依赖
3. github新建库主分支会命名为main，本地git init新建的分支是master，上传的时候会对不上。如果github上有readme和gitignore，由于完全不同的历史，不能合并。（已将github默认分支改为master）
4. 如果本地与github不同，先git pull同步一下再git push就好ß了
5. git push -f强制上传
6. 安装pytorch的时候 torchvision代表一些封装好的vision model，torchaudio代表一些封装好的audio model

## 2021.12.11
1. 想只输入后引号时，点“会自动补全，删前引号会导致后面的也自动被删，可以按着shift+🔙，就可以只删前引号了

## 2021.12.14
1. 注意nn.Module一定要初始化，init和forward完全是两套参数

## 2021.12.15
1. cdf是概率分布函数P(x<=input)，pdf是概率密度函数P(x=input)。pdf是cdf的导数，cdf是pdf的积分。

## 2021.12.16
1. torch.nn.Softmax() 默认dim=0
2. vscode mac 代码格式化快捷键 option+shift+F

## 2021.12.17
1. $a[1,1,1] = a[1][1][1], a[[1,1,1]] = [a[1],a[1],a[1]]$
2. 不是参数的tensor，不需要grad

## 2021.12.18
1. bash判断文件/目录存不存在
    ~~~bash
    dir= ./xx/XXX/
    if [ ! -d ${dir} ];then
        mkdir ${dir}
    fi
    ~~~
    注意”\[“不能与if直接相连
2. 很多tensor自身的操作实际上并不改变自身
   ~~~python
   a.unsqueeze(1)  
   print(a)  # 自身并不改变
   
   a=a.unsqueeze(1)  # 这样才是正确的
   ~~~
3. yapf不加Args或者语法错误会不能生效,注意column_limit两边的{}
   ~~~json
   "python.formatting.yapfArgs": [
        "--style",
        "{column_limit: 256}"
    ],
   ~~~
4. .cuda()和to(device)这种，可以不在class里面加，最后整个model直接to(device)就好


## 2021.12.19
1. 忽略flake8一些烦人的warning：在setting.json里加上
   ~~~json
   "python.linting.flake8Args": [
        "--ignore= F401,E501"
    ],
   ~~~   
2. 如果文件放在一个python包下面，相互引用的时候不能直接文件名，要 
   ~~~python
   from 包名.文件名 import XXX
   ~~~
3. 如果传入的是数字，记得写type
   ~~~python
   parser.add_argument('--num_experts', required=False, type=int, default=16, help="the num of experts")
   ~~~
   不写type就是默认string了，后面不方便
4. 将size的某一维度作为参数传入的时候，用“()”，不是“[]”
   ~~~python
    x1_moe_features, x1_moe_loss = self.Solu_MoE(x1_moe_features, mask1.size[-1], train=self.training)
    TypeError: 'builtin_function_or_method' object is not subscriptable
    # 将size[]->size()
    x1_moe_features, x1_moe_loss = self.Solu_MoE(x1_moe_features, mask1.size(-1), train=self.training)
   ~~~
5. 除了forward的参数可以不用.to(input.device),其它包括init里的参数都需要

## 2021.12.20
1. git 回滚。回滚只会更改它保存的更改，不是直接大版本回去
   回滚之前记得先把现在的add了……
   ~~~bash
   git log                       # 查看历史消息
   git reset --hard HEAD@{$num}  # 跳到num版本
   git reset --hard xxxxx        # 跳到xxxxx版本号
   git reflog                    # 所有git操作
   ~~~
2. ~~~python
   x=x.to(device) # 不要忘了=，to()并不改变自身
   ~~~

## 2021.12.21
1. 调用python往里面传参数的时候，sys.argv[0]是python文件本身，sys.argv[1]开始才是后面加的参数
   ~~~bash
   python test.py 1 2
   # sys.argv[0]=test.py
   # sys.argv[1]=1
   # sys.argv[2]=2
   ~~~
2. bash的if<br>
   一、条件测试的表达式：<br>
    [ expression ]  括号两端必须要有空格<br>
    [[ expression ]] 括号两端必须要有空格<br>

    组合测试条件：<br>
     -a: and<br>
     -o: or<br>
      !: 非

   二、整数比较：<br>
    -eq 测试两个整数是否相等<br>
    -ne 测试两个整数是否不等<br>
    -gt 测试一个数是否大于另一个数<br>
    -lt 测试一个数是否小于另一个数<br>
    -ge 大于或等于<br>
    -le 小于或等于<br>

   命令间的逻辑关系<br>
    逻辑与：&&<br>
    逻辑或：||

   三、字符串比较<br>
      字符串比较：<br>
         ==    等于  两边要有空格<br>
         !=    不等<br>
         >    大于<br>
         <    小于<br>

   四、文件测试

    -z string 测试指定字符是否为空，空着真，非空为假<br>
    -n string 测试指定字符串是否为不空，空为假 非空为真<br>
    -e FILE 测试文件是否存在<br>
    -f file 测试文件是否为普通文件<br>
    -d file 测试指定路径是否为目录<br>
    -r file 测试文件对当前用户是否可读<br>
    -w file 测试文件对当前用户是否可写<br>
    -x file 测试文件对当前用户是都可执行<br>
    -z  是否为空  为空则为真<br>
    -a  是否不空

   五、if语法

   if 判断条件 0为真 其他都为假<br>

   单分支if语句<br>
   if 判断条件；then<br>
   statement1<br>
   statement2<br>
   .......<br>
   fi<br>

   双分支的if语句：<br>
   if 判断条件；then<br>
   statement1<br>
   statement2<br>
   .....<br>
   else<br>
   statement3<br>
   statement4<br>
   fi

## 2021.12.30
1. sftp.json里passphare设置为true，这样就可以每次登录的时候输ssh的密码了，或者直接
   ~~~json
   passphare = "XXXXXX"
   ~~~

## 2021.12.31
1. python format 代码的时候yapf与black的Args不通用
   ~~~json
   "python.formatting.yapfArgs": [
        "--style",
        "{column_limit: 256}"
    ],
   ~~~
   只适用于yapf，用black会导致根本不输入
2. 对于tensor操作里面有dim的，output的维度就是input删去对应dim。可以用这来对照一下结果维度是不是符合自己要求来辨别dim设置的对不对。
>举例:input是3x4x5，<br>
dim=1, output:3x5；<br>
dim=0, output:4x5；<br>
dim=-1, output:3x4.
3. code-runner设置运行python的环境（不要默认的），在setting.json里面executorMap直接加python就行
   ~~~json
   "code-runner.executorMap": {
        "c": "gcc $fileName -o build/$fileNameWithoutExt && ./build/$fileNameWithoutExt",
        "cpp": "clang++ -std=c++17 $fileName -o build/$fileNameWithoutExt && ./build/$fileNameWithoutExt",
        "python": "/Users/s1mple/miniconda3/envs/pytorch/bin/python"
    },
   ~~~
   python跟c，cpp还是不同的，不需要那么详细

## 2022.1.1
1. Tensor.size是function，size(0),size(1)...
   Tensor.shape是attribute，shape=[size(0),size[1]...]
2. 大量输出中间变量的时候，tensor.tolist()之后，round()一下，要不然小数太多