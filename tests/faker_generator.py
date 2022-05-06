import random
import pandas as pd
from faker.providers import BaseProvider
from faker.providers import DynamicProvider
from faker import Faker


binary_provider = DynamicProvider(provider_name="bin", elements=[0, 1])

age_provider = DynamicProvider(
    provider_name="age",
    elements=list(range(29, 78, 1)),
)

cp_provider = DynamicProvider(provider_name="cp", elements=[0, 1, 2, 3])

trestbps_provider = DynamicProvider(
    provider_name="trestbps", elements=list(range(94, 200, 1))
)

chol_provider = DynamicProvider(provider_name="chol", elements=list(range(126, 564, 1)))

restecg_provider = DynamicProvider(provider_name="restecg", elements=[0, 1, 2])

thalach_provider = DynamicProvider(
    provider_name="thalach", elements=list(range(71, 202, 1))
)


class OldpeakProvider(BaseProvider):
    def oldpeak(self) -> float:
        return round(random.uniform(0, 6.2), 1)


slope_provider = DynamicProvider(provider_name="slope", elements=[0, 1, 2])

ca_provider = DynamicProvider(provider_name="ca", elements=[0, 1, 2, 3])

thal_provider = DynamicProvider(provider_name="thal", elements=[0, 1, 2])


def gen_fake_dataset(n_rows: int, target: bool):
    fake = Faker()
    fake.add_provider(binary_provider)
    fake.add_provider(age_provider)
    fake.add_provider(cp_provider)
    fake.add_provider(trestbps_provider)
    fake.add_provider(chol_provider)
    fake.add_provider(restecg_provider)
    fake.add_provider(thalach_provider)
    fake.add_provider(OldpeakProvider)
    fake.add_provider(slope_provider)
    fake.add_provider(ca_provider)
    fake.add_provider(thal_provider)

    dataset = dict()
    dataset["age"] = [fake.age() for _ in range(n_rows)]
    dataset["sex"] = [fake.bin() for _ in range(n_rows)]
    dataset["cp"] = [fake.cp() for _ in range(n_rows)]
    dataset["trestbps"] = [fake.trestbps() for _ in range(n_rows)]
    dataset["chol"] = [fake.chol() for _ in range(n_rows)]
    dataset["fbs"] = [fake.bin() for _ in range(n_rows)]
    dataset["restecg"] = [fake.restecg() for _ in range(n_rows)]
    dataset["thalach"] = [fake.thalach() for _ in range(n_rows)]
    dataset["exang"] = [fake.bin() for _ in range(n_rows)]
    dataset["oldpeak"] = [fake.oldpeak() for _ in range(n_rows)]
    dataset["slope"] = [fake.slope() for _ in range(n_rows)]
    dataset["ca"] = [fake.ca() for _ in range(n_rows)]
    dataset["thal"] = [fake.thal() for _ in range(n_rows)]

    if target:
        dataset["condition"] = [fake.bin() for _ in range(n_rows)]

    return pd.DataFrame(dataset)
