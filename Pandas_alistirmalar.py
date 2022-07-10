# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
import pandas as pd
import seaborn as sns
df=sns.load_dataset("titanic")
df.head()

# Görev 2: Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.

df["sex"].value_counts()

# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.

df.nunique()

# Görev 4: pclass değişkeninin unique değerlerinin sayısını bulunuz.

df["pclass"].nunique()

# Görev 5: pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.

wanted=["pclass","parch"]

df[wanted].nunique()

# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.

df["embarked"].dtype

df["embarked"]=df["embarked"].astype("category")
df["embarked"].dtype


# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.

df.loc[df["embarked"]=="C"]

# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.

df.loc[df["embarked"]!="S"]

# Görev 9: Yaşı 30'dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.

df[(df["age"]<30) & (df["sex"]=="female")]

# Görev 10: Fare'i 500'den büyük veya yaşı 70’den büyük yolcuların bilgilerini gösteriniz.

df[(df["fare"] > 500) | (df["age"] > 70)]

# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.

df.isnull().sum()

# Görev 12: who değişkenini dataframe’den çıkarınız.

df.drop("who",axis=1,inplace=True)

"who" in df # let's check

# Görev 13: deck değişkenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
df["deck"].mode()
type(df["deck"].mode()) #pandas series

df["deck"].fillna(df["deck"].mode()[0], inplace=True)

df.isnull().sum() #checking

# Görev 14: age değişkenindeki boş değerleri age değişkenin medyanı ile doldurunuz.

df["age"].median() #returns integer
df["age"].fillna(df["age"].median(), inplace=True)
df.age.isnull().sum() #checking


# Görev 15: survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.

df.groupby(["pclass","sex"]).agg({"survived" : ["sum","count","mean"]})
df.pivot_table(values="survived",index=["pclass","sex"],aggfunc=["sum","count","mean"])

# Görev 16: 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 verecek bir fonksiyon yazın.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)

df["age"]
df["age"][8]
type(df["age"])

#first way

df["age_flag"]=df["age"].apply(lambda x: 1 if x <30 else(0 if x>=30 else x))

# second way
def condition(x):
    if x<30:
        return 1
    elif x>=30:
        return 0
    else:
        return x

df["age_flag"]=df["age"].apply(condition)
df.tail(15)

# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.

dff=sns.load_dataset("tips")
dff.head()

# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerinin sum, min, max ve mean değerlerini bulunuz.

dff.groupby("time").agg({"total_bill":["sum","min","max","mean"]})

# Görev 19: Day ve time’a göre total_bill değerlerinin sum, min, max ve mean değerlerini bulunuz.

dff.groupby(["time","day"]).agg({"total_bill":["sum","min","max","mean"]})

# Görev 20: Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre sum, min, max ve mean değerlerini bulunuz.

dff[(dff["time"] == "Lunch") & (dff["sex"] == "Female")].groupby("day").agg(
    {"total_bill": ["sum", "min", "max", "mean"], "tip": ["sum", "min", "max", "mean"]}).reset_index()

type(dff[(dff["time"]=="Lunch") & (dff["sex"]=="Female")])


# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)

dff.loc[(dff["size"]<3) & (dff["total_bill"]>10)]["total_bill"].mean()

# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturunuz. Her bir müşterinin ödediği total_bill ve tip in toplamını versin.

dff["total_bill_tip_sum"]=dff["total_bill"]+dff["tip"]
dff.head()

# Görev 23: Total_bill değişkeninin kadın ve erkek için ayrı ayrı ortalamasını bulunuz.
# Bulduğunuz ortalamaların altında olanlara 0, üstünde ve eşit olanlara 1 verildiği yeni bir total_bill_flag değişkeni oluşturunuz.
# Kadınlar için Female olanlarının ortalamaları, erkekler için ise Male olanların ortalamaları dikkate alınacaktır.
# Parametre olarak cinsiyet ve total_bill alan bir fonksiyon yazarak başlayınız. (If-else koşulları içerecek)

dff_means=dff.groupby("sex")["total_bill"].mean()
dff_means.head()
dff_means
type(dff_means["total_bill"])

def male_female_mean(cinsiyet,total_bill):
        if cinsiyet=="Male" and total_bill >= dff_means["Male"]:
            return 1
        elif cinsiyet=="Female" and total_bill >= dff_means["Female"]:
            return 1
         else:
            return 0
male_female_mean("Male",300)
# loop
totbillflag=[]
for (cins,tot) in zip(dff["sex"], dff["total_bill"]):
        totbillflag.append(male_female_mean(cins,tot))

len(totbillflag)
dff["total_bill_flag"]= totbillflag
dff.head()

# comprehension
dff["total_bill_flag"]=[male_female_mean(cins,tot) for (cins,tot) in zip(dff["sex"], dff["total_bill"])]
dff.head()

# Görev 24: total_bill_flag değişkenini kullanarak cinsiyetlere göre ortalamanın altında ve üstünde olanların sayısını gözlemleyiniz.

dff.groupby(["sex"])["total_bill_flag"].value_counts()

# Görev 25: Veriyi total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.

dff_sorted_thirty=dff.sort_values("total_bill_tip_sum")[0:30]
dff_sorted_thirty
