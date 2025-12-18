# Laborator 08 — Machine Learning in Bioinformatica

## Tipuri de invatare
In acest laborator au fost explorate trei tipuri principale de Machine Learning:

- **Invatare supravegheata (Supervised Learning)**, unde fiecare proba are o eticheta cunoscuta.
  A fost folosita pentru clasificarea probelor biologice folosind Random Forest si Logistic Regression.

- **Invatare nesupravegheata (Unsupervised Learning)**, unde etichetele nu sunt cunoscute.
  A fost utilizata metoda KMeans pentru a identifica clustere naturale in datele de expresie genica.

- **Invatare semi-supravegheata**, discutata conceptual, ca scenariu in care doar o parte din date
  sunt etichetate, situatie frecventa in bioinformatica reala.

## Datele folosite
A fost utilizata o matrice de expresie genica sintetica, cu 50 de gene si 180 de probe,
impartite in trei clase biologice:
- Control
- CancerA
- CancerB

Datele au fost construite astfel incat anumite gene sa fie mai expresate in clasele de cancer,
simuland un scenariu biologic realist.

## Supervised Learning — Random Forest
Modelul Random Forest a fost antrenat pentru clasificarea probelor pe baza expresiei genice.
Performanta obtinuta a fost foarte ridicata (accuracy = 1.00), ceea ce indica o separare clara
intre clase in acest set de date.

Au fost generate:
- raportul de clasificare,
- matricea de confuzie,
- importanta genelor (feature importance).

Genele cu importanta mare pot fi interpretate ca potentiali biomarkeri.

## Comparatie Random Forest vs Logistic Regression
A fost realizata o comparatie intre un model non-liniar (Random Forest) si un model liniar
(Logistic Regression).

Random Forest a capturat mai bine relatiile complexe dintre gene, in timp ce Logistic Regression
ofera un model mai simplu si mai usor de interpretat.
Comparatia evidentiaza avantajele si limitarile fiecarui tip de model.

## Unsupervised Learning — KMeans
KMeans a fost aplicat fara a folosi etichetele reale, pentru a observa daca datele se grupeaza
natural.
Crosstab-ul dintre clustere si etichetele reale a aratat o corespondenta buna, confirmand
structura clara a datelor.

## Concluzie
Machine Learning este un instrument esential in analiza datelor omice, permitand:
- clasificarea probelor biologice,
- descoperirea de biomarkeri,
- intelegerea structurii ascunse a datelor.

Rezultatele obtinute demonstreaza diferentele dintre metodele supravegheate si nesupravegheate
si importanta alegerii corecte a modelului in functie de problema biologica.
