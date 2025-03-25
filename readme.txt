✅ 你成功运行 Docker 容器后，会进入一个干净的 Linux shell，下一步就该：
✅ 步骤 1：安装你的训练所需依赖（在容器内部运行）
bash
Copy
Edit
pip install numpy opencv-python-headless ale-py imageio
如果你有 requirements.txt 文件：

bash
Copy
Edit
pip install -r requirements.txt
✅ 步骤 2：确保你把项目代码挂载进来了（最好）
你启动容器时，像这样挂载本地代码：

bash
Copy
Edit
docker run --gpus all -it \
  -v $(pwd):/workspace \
  -w /workspace \
  tensorflow/tensorflow:2.13.0-gpu bash
-v $(pwd):/workspace：把你当前目录挂载进容器的 /workspace

-w /workspace：一进容器就定位到 /workspace

🧠 所以你进去之后直接可以：
bash
Copy
Edit
ls
应该能看到你的代码，比如 main.py、emulator.py 等。

✅ 步骤 3：运行你的训练脚本
直接运行你以前的命令即可：

bash
Copy
Edit
python main.py
✅ 步骤 4（可选）：保存训练 checkpoint/model
TensorFlow 会保存模型到类似：

bash
Copy
Edit
ckpt/model_10000
这些文件会存在 /workspace/ckpt（挂载的本地目录），所以不会丢失，退出容器后你本地仍然可以看到。

✅ 步骤 5：退出容器
输入：

bash
Copy
Edit
exit
即可退出容器，回到你宿主机的终端。

✅ 想再次进入容器？
你可以重新创建，或者用 容器名 启动：

bash
Copy
Edit
# 查看历史容器
docker ps -a

# 启动已存在的容器（CONTAINER_ID 改成你自己的）
docker start -ai <CONTAINER_ID>







启动docker: docker start unruffled_jepsen
进入docker:
 docker exec -it unruffled_jepsen bash

unruffled_jepsen