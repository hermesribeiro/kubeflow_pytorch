from kfp import dsl
from kfp.aws import use_aws_secret

from helpers.kfp_auth import KfpAuth


@dsl.pipeline(name="CIFAR Pytorch", description="hello world")
def cifar_pipeline(
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    bucket: str,
    path: str,
):
    train_op = (
        dsl.ContainerOp(
            name="Model Train",
            image="hermesribeiro/cifar:latest",
            command=["python", "train.py"],
            arguments=[
                "-n",
                num_epochs,
                "-b",
                batch_size,
                "-l",
                learning_rate,
                "-m",
                momentum,
                "-u",
                bucket,
                "-p",
                path,
            ],
        )
        .apply(use_aws_secret())
        .set_image_pull_policy("Always")
        .set_memory_request('2G')
        .set_cpu_request('4')
        # .set_gpu_limit('1', 'nvidia')
        # .add_volume_mount(...)
        # .add_env_variable(V1EnvVar(name='HOST', value='foo.bar'))
        # .set_retry(10)
    )

    eval_op = (
        dsl.ContainerOp(
            name="Model Eval",
            image="hermesribeiro/cifar:latest",
            command=["python", "eval.py"],
            arguments=["-u", bucket, "-p", path],
        )
        .apply(use_aws_secret())
        .set_image_pull_policy("Always")
        .after(train_op)
    )


if __name__ == "__main__":
    client = KfpAuth().client()
    client.create_run_from_pipeline_func(
        cifar_pipeline,
        arguments={
            "num_epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.001,
            "momentum": 0.0,
            "bucket": "hermes-freestyle",
            "path": "cifar/cifar_net.pth",
        }
    )
