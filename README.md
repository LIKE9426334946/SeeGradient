# SeeGradient

一个用于学习全连接神经网络的梯度可视化网页。拖动权重柱状图，实时查看预测值、均方误差（MSE）以及各个参数的梯度。

## 功能

- 设置输入矩阵 X、目标矩阵 Y、隐藏层形状、激活函数及偏置开关。
- 支持没有隐藏层的矩阵乘法，以及 1–4 个隐藏层；每层 1–8 个神经元、1–16 个样本。
- X、Y 支持逗号/空格分列、换行/分号分行，也支持二维 JSON 数组。
- 每层权重 W 和偏置 b 都能查看。W 为 2 × 2 时，画布上显示 2 行 2 列的四根数值柱；右侧梯度画布位置与参数完全对应。
- 拖动柱子或柱顶圆点改变参数，允许正负值；按住 Shift 可精细拖动，也支持精确数字输入和键盘操作。
- 实时计算整个网络的前向传播、损失和反向传播。输出层始终为线性层。
- 可调学习率、单步 SGD 更新、连续训练、三档训练速度、单次撤销、重新初始化。
- 实时预测/目标/平方误差表格、最近 240 次损失变化曲线。
- 入门示例和纯矩阵乘法示例；浅色/深色主题；响应式布局。
- 当前实验自动存入当前浏览器的 localStorage，刷新继续。**不是服务器存档，不跨浏览器或设备同步。**没有数据库、账号或登录。
- 无第三方运行时依赖、无前端构建步骤、无外部 CDN；部署后页面不需要访问其他网站。

## 计算约定

按行存放样本，`X` 为 `[B, in]`，`W` 为 `[in, out]`，`b` 为 `[1, out]`：

```text
Z = A @ W + b
A = activation(Z)           # 隐藏层
Y_hat = Z                   # 最后一层，不使用隐藏层激活函数
L = sum((Y_hat - Y)^2) / (B * output_dim)
```

这里 W 的存储方式是 **输入维度 × 输出维度**；PyTorch 的 `nn.Linear.weight` 是 **输出维度 × 输入维度**，两者互为转置。

```text
dL/dY_hat = 2 * (Y_hat - Y) / (B * output_dim)
dZ = dA * activation'(Z)    # 最后一层导数为 1
dW = A_prev.T @ dZ
db = sum(dZ, axis=0)
dA_prev = dZ @ W.T
theta_new = theta_old - learning_rate * gradient
```

ReLU 在 0 处采用导数 0；Sigmoid 使用稳定计算方式。MSE 没有额外的 `1/2` 系数。每次更新使用同一次反向传播得到的梯度，同时更新全部参数；拖动则仅改变选中的单个参数。

梯度反映当前位置的一阶局部变化率，不能保证大幅拖动或使用较大学习率时损失下降。梯度接近零也不保证全局最优。图中参数与梯度**分别采用自己的统一纵轴刻度**，以页脚的 `±数值` 为准，不能直接比较两幅图的柱高。

拖动过程中固定参数刻度，避免柱子随缩放跳动；松手后自动留出刻度余量。可反复拖动或精确输入更大的值。X/Y 绝对值最多 1000，参数绝对值最多 1,000,000；溢出的更新会整体拒绝并保留原模型。连续训练会在数值异常、页面隐藏、损失小于 1e-12 或梯度范数小于 1e-12 时停止。

## 操作

1. 初次打开已有一个 `2 → 2 → 1` 网络。先拖动 W₁ 中的一根柱子，观察 MSE 和右侧梯度。
2. 点击柱子后，可在下方“当前参数”输入精确数字。索引从 0 开始。
3. 点击层标签切换 W₁/W₂ 等参数，点击“偏置 b”切换到偏置。
4. 点击“单步更新”执行一次完整 SGD，或点击“连续训练”观察损失变化；再次点击可暂停。
5. 修改左侧 X/Y/隐藏层后，点击“应用网络设置”。仅改变数据或激活函数时保留参数；形状或偏置开关改变时重新初始化。重新应用设置会清空训练步数和损失记录。
6. “撤销”回退上一次手动编辑、初始化、载入示例、单步训练，或整段连续训练。一次拖动视为一次操作，仅保留一个撤销快照。

键盘：Tab 聚焦参数画布；←/→ 选择元素；↑/↓ 调整 0.01；按住 Shift 调整 0.001；Home 归零；Enter 聚焦精确输入框；拖动时 Esc 取消。梯度图只读，可用方向键选择对应参数。

## 项目结构

```text
SeeGradient/
  backend/server.js              Node.js 服务，只监听回环地址
  public/index.html              页面
  public/styles.css              响应式样式与主题
  public/app.js                  状态与交互
  public/math.js                 前向传播、解析梯度、SGD
  public/charts.js               Canvas 柱状图与损失曲线
  public/favicon.svg             项目标识
  deploy/SeeGradient.service     systemd 服务配置
  deploy/SeeGradient.nginx.conf  Nginx 反向代理配置
  tests/                        数值与服务测试
```

## Ubuntu 单服务器部署

使用 root；项目目录固定为 `/opt/SeeGradient`。需要 `/usr/bin/node` 为 **Node.js 22 或更高版本**、Git、Nginx。无 npm 依赖需要安装。

端口关系：

| 入口 | 地址 |
| --- | --- |
| 公网 Nginx | `http://服务器IP:16048` |
| 内部 Node.js | `127.0.0.1:3048` |

### 1. 准备环境并拉取代码

```bash
apt update
apt install -y git nginx curl ca-certificates
/usr/bin/node --version
mkdir -p /opt/SeeGradient
git clone -b main https://github.com/LIKE9426334946/SeeGradient.git /opt/SeeGradient
cd /opt/SeeGradient
```

如果 `/usr/bin/node` 不存在或版本低于 22，可先按 [NodeSource 的 Node.js 22 安装方式](https://github.com/nodesource/distributions/blob/master/DEV_README.md) 安装，再继续后面的步骤：

```bash
curl -fsSL https://deb.nodesource.com/setup_22.x -o /tmp/seegradient-node-setup.sh
bash /tmp/seegradient-node-setup.sh
apt install -y nodejs
/usr/bin/node --version
```

服务使用固定路径 `/usr/bin/node`，不会自动读取交互式 shell 的 nvm 环境。已有满足要求的 Node.js 时不用重复安装。

### 2. 创建 systemd 服务文件

仓库中的服务配置已包含 root、工作目录、Node 入口、回环监听、内部端口及自动重启：

```bash
cp /opt/SeeGradient/deploy/SeeGradient.service /etc/systemd/system/SeeGradient.service
systemctl daemon-reload
systemctl enable SeeGradient
systemctl start SeeGradient
systemctl status SeeGradient --no-pager
```

内部健康检查：

```bash
curl http://127.0.0.1:3048/api/health
```

正常返回 `{"status":"ok","app":"SeeGradient"}`。

### 3. 创建并启用 Nginx 独立配置

```bash
cp /opt/SeeGradient/deploy/SeeGradient.nginx.conf /etc/nginx/sites-available/SeeGradient
ln -sfn /etc/nginx/sites-available/SeeGradient /etc/nginx/sites-enabled/SeeGradient
nginx -t
```

确认 `nginx -t` 成功后：

```bash
systemctl enable nginx
systemctl restart nginx
```

此配置使用独立 `server` 块监听 16048，转发到 `127.0.0.1:3048`，含 Host、真实 IP、转发协议和 WebSocket 请求头。应用当前不使用 WebSocket，但反向代理配置支持后续添加。

### 4. 开放外部端口并访问

在云服务器安全组中允许 **TCP 16048**。如果已启用 UFW：

```bash
ufw allow 16048/tcp
```

内部 3048 不需要开放。浏览器访问：

```text
http://你的服务器IP:16048
```

### 更新代码

```bash
cd /opt/SeeGradient
git pull --ff-only origin main
systemctl restart SeeGradient
```

如果 Nginx 或 systemd 配置发生变化，重新复制相应配置并执行 `nginx -t` / `systemctl daemon-reload` 后重启服务。更新后刷新网页；如旧资源仍被缓存，使用 Ctrl+F5。

### 排查

```bash
journalctl -u SeeGradient -n 100 --no-pager
systemctl status SeeGradient --no-pager
nginx -t
curl http://127.0.0.1:16048/api/health
ss -lntp
```

Node 仅允许 `HOST=127.0.0.1` 或 `::1`，防止误将内部服务暴露到公网。静态服务使用文件白名单，不提供仓库和服务器目录访问。

## 本地运行与验证

```bash
node backend/server.js
```

打开 `http://127.0.0.1:3048`。代码使用浏览器 ES Modules，需要通过 HTTP 服务打开，不能直接双击 HTML。

```bash
npm run check
npm test
```

测试包括手算矩阵样例、多输出 MSE 归一化、各激活函数和多层网络的中心差分梯度检查、输入梯度、同步参数更新、溢出回滚、输入校验、拖动值映射以及静态资源与健康检查。无需下载测试依赖。
