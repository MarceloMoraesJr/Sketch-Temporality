FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

COPY requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt

RUN apt-get update && \
    apt-get install -y git tmux wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["tail"]
CMD ["-f","/dev/null"]
