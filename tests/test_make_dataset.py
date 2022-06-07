from ml_project.data import split_data


def test_splitting(dataset, splitting_params):
    df_train, df_test = split_data(dataset, splitting_params, "condition")
    assert len(df_train) == len(df_test)
    assert (
        abs(
            df_train["condition"].value_counts()[1]
            - df_test["condition"].value_counts()[1]
        )
        <= 1
    )
