# VolGAN-BR (starter)
Replicar **Vuletić & Cont (2024/2025)** (*VolGAN*) no Brasil, usando opções da **B3** (Ibovespa/BOVA11).

## Estrutura
```
volgan_b3_starter/
  data/
    raw/         # coloque aqui os arquivos COTAHIST/Opções da B3 (.txt/.zip)
    processed/   # saídas intermediárias (parquet/csv)
  src/
    data_b3.py   # leitor/parse para COTAHIST/Opções
    iv_utils.py  # utilitários de IV, forward e VXBR-like
    surface.py   # montagem/limpeza da “grade” IV (k,T)
  notebooks/
    00_quickstart.ipynb  # roteiro com células para testar no seu ambiente local
```

## Passos (visão geral)
1) **Baixar** histórico de **opções sobre Ibovespa** (ou **BOVA11**) no site da B3 (COTAHIST).  
2) **Parsear** (src/data_b3.py) → DataFrame (data, ticker_underlying, tipo C/P, strike, vencimento, preço, qtd negócios).  
3) **Estimativa do forward** (src/iv_utils.py) via **put–call parity** por vencimento (minimização NLS), e cálculo de **IV** por opção (BS/Black).  
4) **Construir superfícies IV**: coordenadas padronizadas **k=ln(K/F)** e **T** em anos; filtrar illiquid/OTM extremos; salvar grade diária (src/surface.py).  
5) **Treino do gerador** (VolGAN) → aqui você pluga seu modelo (PyTorch) usando os grids diários.
6) **Validação**: simular séries e comparar com o **VXBR** (índice de vol implícita de 30d do Ibovespa) e com estatísticas de co-movimento (autocorrelação, PCA dos modos de variação, etc.).

> Observação: os scripts **não** fazem download automático (o site da B3 costuma exigir captchas/mudanças de layout). Use **rb3/finbr** no R/Python ou baixe manualmente e depois **parseie localmente** com `data_b3.py`.
