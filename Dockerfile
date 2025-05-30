FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN pip install lightning
RUN apt-get update && \
    apt-get install -y git tmux && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["tail"]
CMD ["-f","/dev/null"]
