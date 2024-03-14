import mlflow

from house_price_prediction.score import evaluate_model
from house_price_prediction.train import train_model
from house_price_prediction.ingest_data import fetch_housing_data


def main():
    remote_server_url = "http://0.0.0.0:5001"
    mlflow.set_tracking_uri(remote_server_url)
    tracking_uri = mlflow.tracking.get_tracking_uri()
    print(f"Current tracking uri: {tracking_uri}")

    experiment_name = "house_price_prediction"

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(
        run_name="Main_function",
        experiment_id=experiment_id,
        tags={"version": "v1", "priority": "P1"},
        description="parent",
    ) as parent_run:
        mlflow.log_param("parent", "yes")
        with mlflow.start_run(
            run_name="get_data",
            experiment_id=experiment_id,
            description="child",
            nested=True,
        ):
            mlflow.log_param("child", "yes")
            fetch_housing_data("../data/housing")
            mlflow.log_param("output_folder", "../data/housing")
        with mlflow.start_run(
            run_name="train_data",
            experiment_id=experiment_id,
            description="child",
            nested=True,
        ):
            mlflow.log_param("child", "yes")
            train_model("../data/housing", "../model")
            mlflow.log_artifact("../logs/score_data.log")
        with mlflow.start_run(
            run_name="score",
            experiment_id=experiment_id,
            description="child",
            nested=True,
        ):
            mlflow.log_param("child", "yes")
            temp1 = evaluate_model(
                "../model/trained_model.pkl",
                "../data/housing",
                experiment_name,
                remote_server_url,
            )
            mlflow.log_artifact("../logs/train_data.log")
            mlflow.sklearn.log_model("../model/trained_model.pkl", "trained_model")

    query = f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'"
    results = mlflow.search_runs(experiment_ids=[experiment_id], filter_string=query)
    print("child runs:")
    print(results[["run_id", "params.child", "tags.mlflow.runName"]])


if __name__ == "__main__":
    main()
