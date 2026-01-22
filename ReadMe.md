# 🧠 Tarixiy Shaxslarning Kasbiga Ko‘ra Umrini Bashoratlash — Regression Machine Learning Loyiha

## 📌 Loyihaning qisqacha tavsifi

Ushbu loyiha tarixiy shaxslarning kasbi va boshqa atributlari asosida ularning umr davomiyligini **doimiy son (yillar)** sifatida bashorat qilishga qaratilgan. Maqsad — qaysi kasb egalari uzoq umr ko‘rganini aniqlash va bashorat qilish.

Datasetda 2500 dan ortiq tarixiy shaxs haqidagi ma’lumotlar mavjud bo‘lib, ularning tug‘ilgan va vafot etgan yili, kasbi, yashagan davri kabi atributlar kiritilgan.

---

## 🚀 Technical Contribution

### 📊 Project Overview
- Ushbu loyiha tarixiy va biografik ma’lumotlar asosida shaxslarning **umr davomiyligini (`life_span`)** bashorat qilishga qaratilgan **supervised regression** muammosini hal qiladi.
- Dataset **2022 ta kuzatuv** va **20 ta feature** dan iborat bo‘lib, demografik, biografik hamda **cluster-based** ustunlarni o‘z ichiga oladi.

### 🧹 Data Processing & Feature Engineering
- Ma’lumotlarda **missing value** mavjud emas, bu esa model barqarorligini oshirdi.
- Yuqori kardinalilikka ega kategorik ustunlar (`occupation`, `education`, `awards`) **clustering va encoding** orqali siqildi.
- Sana ustunlaridan (`birth_date`, `death_date`) **numeric yilga oid feature** lar ajratildi.
- Modelga mos bo‘lishi uchun faqat **numeric va encoded feature** lar ishlatildi.

### 📈 Modeling & Evaluation
- Baseline sifatida **Linear Regression, Decision Tree, Random Forest va XGBoost** modellari baholandi.
- Linear Regression past natija ko‘rsatib, ma’lumotlardagi **non-linear bog‘lanishlarni** ushlay olmasligi aniqlandi.
- Ensemble modellar ichida **Gradient Boosting** eng yaxshi natijani berdi  
  👉 **R² = 0.8426**, **RMSE = 30.33**.
- Ushbu model **bias–variance balansini** samarali ushlagani uchun asosiy model sifatida tanlandi.

### ⚙️ Hyperparameter Optimization
- Tanlangan Gradient Boosting modeli **Optuna** yordamida tuning qilindi.
- `n_estimators`, `learning_rate`, `max_depth`, `subsample` kabi parametrlar optimallashtirildi.
- Tuning jarayonida **overfittingni kamaytirish** va **generalization** ni yaxshilash maqsad qilindi.

### 🧠 Final Outcome
- Loyiha davomida **tree-based va ensemble modeling** strategiyalari muvaffaqiyatli qo‘llandi.
- Feature engineering va model tanlash **tizimli va asosli** tarzda amalga oshirildi.
- Natijada, loyiha **ishlab chiqarishga tayyor**, **barqaror** va **tushuntiriladigan** regression pipeline bilan yakunlandi ✅




## ⚙️ Ishlatilgan texnologiyalar

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn (vizualizatsiya uchun)
- Jupyter Notebook

---

## 💼 Business Contribution

### 🎯 Business Objective
- Loyiha maqsadi — tarixiy va biografik ma’lumotlar asosida shaxslarning **taxminiy umr davomiyligini oldindan bashorat qilish**, bu orqali **analitik qarorlar qabul qilishni qo‘llab-quvvatlash**.

### 📊 Business Value
- Model shaxslarning umr davomiyligiga ta’sir qiluvchi **asosiy omillarni aniqlash** imkonini berdi (ta’lim, kasb, mukofotlar, biografik klasterlar).
- Murakkab ma’lumotlarni **oddiy va tushunarli bashoratga** aylantirib, biznes foydalanuvchilar uchun qiymat yaratildi.
- Bashorat natijalari **tarixiy tadqiqotlar**, **sug‘urta risk tahlili**, **demografik analitika** va **kontent tavsiya tizimlari** uchun ishlatilishi mumkin.

### 🚀 Decision Support
- Ensemble modeling yordamida olingan natijalar **bir modelga tayanish riskini kamaytirdi**.
- Gradient Boosting modeli **eng ishonchli va barqaror yechim** sifatida tanlanib, real muhitda qo‘llashga tayyor holatga keltirildi.
- Model natijalari **scenario-based tahlil** va **prognozlash** uchun foydalanilishi mumkin.

### 💰 Efficiency & Impact
- Feature dimensionality clustering orqali kamaytirilib, **hisoblash resurslari tejaldi**.
- Avtomatlashtirilgan modeling pipeline **vaqt va operatsion xarajatlarni qisqartirdi**.
- Loyihaning modulli tuzilishi uni boshqa dataset va sohalarga **tez moslashtirish imkonini berdi**.

### 🧠 Strategic Outcome
- Data-driven yondashuv asosida **qaror qabul qilish sifati oshdi**.
- Model biznes jamoa uchun **analitik ishonch** va **prognoz aniqligini** ta’minladi.
- Natijada loyiha **uzoq muddatli strategik rejalashtirish** uchun mustahkam analitik asos yaratdi ✅


## 💻 Loyihani ishga tushirish

```bash
📞 Aloqa

Loyihaga oid savollar uchun:
Email: rassiazzi9218@gmail.com

GitHub: https://github.com/rasulbekdeveloper907