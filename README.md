# Repo Masterarbeit Jonas Pamminger

**Author:** Jonas Pamminger
**Course:** ExInt II – Research Designs in SME Research, WU Vienna, SS 2026

## Research Question

[Wie beeinflussen geopolitische Sanktionen die De-
Internationalisierung von MNEs?]

## Hypotheses

- H1: [Wie beeinflussen geopolitische Sanktionen die De-
Internationalisierung von MNEs?]

## Variables

### Dependent variable (Y)

| Construct | Data Item(s) | Formula   | Note |
|-----------|-------------|-----------|------|
| RoA       | IB, AT      | IB / AT   | Return on assets; `ib` = income before extraordinary items |

### Independent variable (X)

| Construct | Data Item(s) | Formula     | Note |
|-----------|-------------|-------------|------|
| DOI (Degree of Internationalization) | PIFO, SALE | PIFO / SALE | Foreign pre-tax income share; proxy for MNE internationalization |

### Controls

| Construct      | Data Item(s)   | Formula               | Note |
|----------------|----------------|-----------------------|------|
| Firm size      | AT             | log(AT)               | Log-transform reduces skewness |
| Leverage       | DLTT, AT       | DLTT / AT             | Long-term debt over total assets |
| Firm age       | FYEAR, INCO    | FYEAR − INCO          | Years since incorporation |
| R&D intensity  | XRD, AT        | XRD / AT              | 0 if XRD missing (firm did not report R&D) |

## Data

| Item         | Detail                            |
|--------------|-----------------------------------|
| Source       | WRDS / Compustat Global           |
| Table        | comp_global_daily.g_funda         |
| Downloaded   | 2026-06-01                        |
| License      | WRDS subscriber agreement         |
| Fiscal years | 2005–2020                         |
| Raw rows     | 89,849 (9,308 firms, 23 countries) |
| Clean rows   | 29,344 (3,843 firms, 23 countries) |
