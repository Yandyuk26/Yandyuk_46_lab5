import pandas as pd
import matplotlib.pyplot as plt

DATA_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)

try:
    df = pd.read_csv("titanic.csv")
except FileNotFoundError:
    df = pd.read_csv(DATA_URL)

df.info()
print()
print(df.head(5))
print(df.tail(10))

df_survived_under18 = df[
    (df["Survived"] == 1) & (df["Age"].notna()) & (df["Age"] < 18)
].copy().reset_index(drop=True)
print(df_survived_under18)

df_cherbourg = df[df["Embarked"] == "C"].copy().reset_index(drop=True)
print(df_cherbourg)


def age_group(age):
    if pd.isna(age):
        return None
    if age <= 12:
        return "діти"
    if age <= 17:
        return "підлітки"
    if age <= 35:
        return "молоді"
    if age <= 59:
        return "працездатний вік"
    return "пенсійний вік"


df_south = df[(df["Embarked"] == "S") & (df["Age"].notna())].copy()
df_south["група"] = df_south["Age"].apply(age_group)
counts = df_south["група"].value_counts()
order = ["діти", "підлітки", "молоді", "працездатний вік", "пенсійний вік"]
counts = counts.reindex(order).fillna(0).astype(int)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(counts)), counts.values, tick_label=counts.index, color="steelblue", edgecolor="black")
ax.set_title("Кількість пасажирів за віком (Саутгемптон, Embarked = S)")
ax.set_xlabel("Вікова група")
ax.set_ylabel("Кількість")
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
fig.tight_layout()
plt.show()
