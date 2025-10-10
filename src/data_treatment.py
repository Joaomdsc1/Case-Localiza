import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def detailed_data_analysis(df):
    """Análise dos dados antes do processamento"""
    print("\n=== ANÁLISE EXPLORATÓRIA ===")
    
    # Informações básicas
    print(f"Shape: {df.shape}")
    print(f"Colunas: {df.columns.tolist()}")
    
    # Valores únicos por coluna categórica
    categorical_cols = ['transaction_type', 'location_region', 'purchase_pattern', 
                       'age_group', 'anomaly']
    
    for col in categorical_cols:
        if col in df.columns:
            unique_vals = df[col].value_counts()
            print(f"\n{col} (Total: {len(unique_vals)} categorias):")
            print(unique_vals.head(10))
            if len(unique_vals) > 10:
                print(f"... e mais {len(unique_vals) - 10} categorias")
    
    # Estatísticas descritivas para colunas numéricas
    numeric_cols = ['amount', 'login_frequency', 'session_duration', 'risk_score']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    if numeric_cols:
        print("\nEstatísticas descritivas (colunas numéricas):")
        # Tentar converter para numérico
        temp_df = df[numeric_cols].copy()
        for col in numeric_cols:
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
        
        # Configurar para não usar notação científica
        with pd.option_context('display.float_format', '{:.2f}'.format):
            print(temp_df.describe())
    
    print("\nVerificação de outliers:")
    for col in numeric_cols:
        if col in df.columns:
            try:
                series = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(series) > 0:
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = series[(series < lower_bound) | (series > upper_bound)].shape[0]
                    print(f"  {col}: {outliers} outliers ({outliers/len(series)*100:.2f}%)")
            except:
                print(f"  {col}: Não foi possível calcular outliers")

def validate_data_quality(df):
    """Validação abrangente da qualidade dos dados"""
    validation_issues = {}
    
    print("\n--- VALIDAÇÃO DE QUALIDADE INICIAL ---")
    
    # 1. Valores ausentes
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if len(missing_data) > 0:
        validation_issues['missing_values'] = missing_data
        print(f"Valores ausentes encontrados: {dict(missing_data)}")
    else:
        print("✓ Nenhum valor ausente encontrado")
    
    # 2. Valores duplicados
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        validation_issues['duplicate_rows'] = duplicate_count
        print(f"Registros duplicados: {duplicate_count}")
    else:
        print("✓ Nenhum registro duplicado encontrado")
    
    # 3. Valores inconsistentes em colunas categóricas
    expected_categories = {
        'transaction_type': ['transfer', 'purchase', 'sale', 'exchange'],
        'location_region': ['North America', 'South America', 'Europe', 'Asia', 'Africa'],
        'purchase_pattern': ['focused', 'high_value', 'random'],
        'age_group': ['new', 'established', 'veteran'],
        'anomaly': ['low_risk', 'moderate_risk', 'high_risk']
    }
    
    categorical_issues = {}
    for col, expected_vals in expected_categories.items():
        if col in df.columns:
            unique_vals = df[col].astype(str).unique()
            invalid_vals = [val for val in unique_vals if val not in expected_vals and val not in ['nan', 'NaN', 'None', 'none']]
            if invalid_vals:
                categorical_issues[col] = invalid_vals
    
    if categorical_issues:
        validation_issues['invalid_categories'] = categorical_issues
        print("Valores categóricos inválidos encontrados:")
        for col, vals in categorical_issues.items():
            print(f"  {col}: {vals}")
    else:
        print("✓ Todos os valores categóricos estão dentro do esperado")
    
    # 4. Valores numéricos fora de faixa esperada
    numeric_checks = {
        'amount': (0, 1000000),
        'login_frequency': (0, 50),
        'session_duration': (0, 1000),
        'risk_score': (0, 100)
    }
    
    range_issues = {}
    for col, (min_val, max_val) in numeric_checks.items():
        if col in df.columns:
            try:
                series = pd.to_numeric(df[col], errors='coerce')
                out_of_range = series[(series < min_val) | (series > max_val)].shape[0]
                if out_of_range > 0:
                    range_issues[col] = out_of_range
            except:
                range_issues[col] = "Erro na conversão"
    
    if range_issues:
        validation_issues['out_of_range_values'] = range_issues
        print("Valores fora da faixa esperada:")
        for col, count in range_issues.items():
            print(f"  {col}: {count}")
    else:
        print("✓ Todos os valores numéricos estão dentro da faixa esperada")
    
    return validation_issues

def enhanced_data_cleaning(df):
    """Limpeza e tratamento mais abrangente dos dados"""
    print("\n--- PROCESSO DE LIMPEZA DETALHADO ---")
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # 1. Tratamento de endereços inválidos
    print("1. Validando endereços...")
    if 'sending_address' in df_clean.columns and 'receiving_address' in df_clean.columns:
        address_mask = (df_clean['sending_address'].str.len() == 42) & \
                       (df_clean['receiving_address'].str.len() == 42) & \
                       (df_clean['sending_address'].str.startswith('0x')) & \
                       (df_clean['receiving_address'].str.startswith('0x'))
        
        invalid_addresses = (~address_mask).sum()
        if invalid_addresses > 0:
            print(f"   Removendo {invalid_addresses} registros com endereços inválidos")
            df_clean = df_clean[address_mask]
    
    # 2. Tratamento de ip_prefix
    print("2. Tratando ip_prefix...")
    if 'ip_prefix' in df_clean.columns:
        df_clean['ip_prefix'] = df_clean['ip_prefix'].astype(str)
        invalid_ips = df_clean['ip_prefix'].isin(['0.0', '0', 'nan', 'NaN', 'None', 'none'])
        invalid_ip_count = invalid_ips.sum()
        if invalid_ip_count > 0:
            print(f"   Encontrados {invalid_ip_count} IPs inválidos")
            # Registros são mantidos, mas com IPs marcados como 'INVALID_IP'
            df_clean.loc[invalid_ips, 'ip_prefix'] = 'INVALID_IP'
    
    # 3. Tratamento de risk_score
    print("3. Tratando risk_score...")
    if 'risk_score' in df_clean.columns:
        # Identificar diferentes representações de valores ausentes
        risk_score_missing = df_clean['risk_score'].isin(['none', 'None', 'NULL', 'null', ''])
        missing_count = risk_score_missing.sum()
        print(f"   Valores ausentes em risk_score: {missing_count}")
        
        # Converter para numérico com tratamento de erros
        df_clean['risk_score'] = pd.to_numeric(df_clean['risk_score'], errors='coerce')
        
        # Preencher valores ausentes com a mediana POR REGIÃO
        if missing_count > 0:
            # Calcular a mediana do risk_score para cada região
            medianas_por_regiao = df_clean.groupby('location_region')['risk_score'].median()
            print(f"   Medianas por região:")
            for regiao, mediana in medianas_por_regiao.items():
                print(f"     {regiao}: {mediana:.2f}")
            
            # Preencher valores ausentes com a mediana da respectiva região
            registros_preenchidos = 0
            for regiao, mediana_regiao in medianas_por_regiao.items():
                # Máscara para registros da região atual com risk_score ausente
                mask = (df_clean['location_region'] == regiao) & (df_clean['risk_score'].isna())
                count_regiao = mask.sum()
                
                if count_regiao > 0:
                    df_clean.loc[mask, 'risk_score'] = mediana_regiao
                    registros_preenchidos += count_regiao
                    print(f"     {regiao}: {count_regiao} valores preenchidos com {mediana_regiao:.2f}")
            
            print(f"   Total preenchidos: {registros_preenchidos} valores ausentes com medianas por região")
            
            # Se ainda houver valores ausentes, caso não tenha região definida, preencher com mediana global
            remaining_missing = df_clean['risk_score'].isna().sum()
            if remaining_missing > 0:
                mediana_global = df_clean['risk_score'].median()
                df_clean['risk_score'] = df_clean['risk_score'].fillna(mediana_global)
                print(f"   {remaining_missing} valores restantes preenchidos com mediana global: {mediana_global:.2f}")
    
    # 4. Tratamento de amount
    print("4. Tratando amount...")
    if 'amount' in df_clean.columns:
        df_clean['amount'] = pd.to_numeric(df_clean['amount'], errors='coerce')
        amount_missing = df_clean['amount'].isna().sum()
        
        if amount_missing > 0:
            print(f"   Valores ausentes em amount: {amount_missing}")
            
            # Calcular a mediana do amount para cada região
            medianas_por_regiao = df_clean.groupby('location_region')['amount'].median()
            print(f"   Medianas por região:")
            for regiao, mediana in medianas_por_regiao.items():
                print(f"     {regiao}: {mediana:.2f}")
            
            # Preencher valores ausentes com a mediana da respectiva região
            registros_preenchidos = 0
            for regiao, mediana_regiao in medianas_por_regiao.items():
                # Máscara para registros da região atual com amount ausente
                mask = (df_clean['location_region'] == regiao) & (df_clean['amount'].isna())
                count_regiao = mask.sum()
                
                if count_regiao > 0:
                    df_clean.loc[mask, 'amount'] = mediana_regiao
                    registros_preenchidos += count_regiao
                    print(f"     {regiao}: {count_regiao} valores preenchidos com {mediana_regiao:.2f}")
            
            print(f"   Total preenchidos: {registros_preenchidos} valores ausentes com medianas por região")
            
            # Se ainda houver valores ausentes, caso não tenha região definida, preencher com mediana global
            remaining_missing = df_clean['amount'].isna().sum()
            if remaining_missing > 0:
                mediana_global = df_clean['amount'].median()
                df_clean['amount'] = df_clean['amount'].fillna(mediana_global)
                print(f"   {remaining_missing} valores restantes preenchidos com mediana global: {mediana_global:.2f}")
        else:
            print("   Nenhum valor ausente encontrado em amount")
    
    # 5. Validação de timestamp
    print("5. Validando timestamps...")
    if 'timestamp' in df_clean.columns:
        df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], unit='s', errors='coerce')
        
        # Verificar timestamps no futuro (possível erro)
        future_timestamps = (df_clean['timestamp'] > pd.Timestamp.now()).sum()
        if future_timestamps > 0:
            print(f"   Atenção: {future_timestamps} timestamps no futuro")
        
        # Verificar timestamps muito antigos
        ancient_timestamps = (df_clean['timestamp'] < pd.Timestamp('2000-01-01')).sum()
        if ancient_timestamps > 0:
            print(f"   Atenção: {ancient_timestamps} timestamps muito antigos")
    
    # 6. Tratamento de location_region
    print("6. Tratando location_region...")
    if 'location_region' in df_clean.columns:
        # Converter todos os registros para minúsculo
        df_clean['location_region'] = df_clean['location_region'].astype(str).str.lower()
        
        # Listar valores únicos após conversão
        unique_regions = df_clean['location_region'].unique()
        print(f"   Regiões únicas após conversão para minúsculo: {unique_regions}")
        
        # Remover registros com valores claramente inválidos (exemplo: '0', 'nan', 'none')
        invalid_regions = ['0', 'nan', 'none', '']
        invalid_count = df_clean['location_region'].isin(invalid_regions).sum()
        if invalid_count > 0:
            print(f"   Regiões inválidas encontradas: {invalid_count}")
            df_clean = df_clean[~df_clean['location_region'].isin(invalid_regions)]
    
    # 7. Remover duplicatas
    print("7. Removendo duplicatas...")
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    after_dedup = len(df_clean)
    duplicates_removed = before_dedup - after_dedup
    if duplicates_removed > 0:
        print(f"   Removidas {duplicates_removed} duplicatas")
    
    final_count = len(df_clean)
    removed_count = initial_count - final_count
    print(f"\n   Registros removidos no total: {removed_count}")
    print(f"   Registros restantes: {final_count} ({final_count/initial_count*100:.2f}% retidos)")
    
    return df_clean

def post_processing_checks(df_processed):
    """Verificações finais após o processamento"""
    print("\n--- VERIFICAÇÕES FINAIS DE CONSISTÊNCIA ---")
    
    checks_passed = 0
    total_checks = 0
    
    # 1. Consistência entre risk_score e anomaly
    if 'risk_score' in df_processed.columns and 'anomaly' in df_processed.columns:
        total_checks += 1
        inconsistency = df_processed[
            (df_processed['risk_score'] > 75) & (df_processed['anomaly'] == 'low_risk') |
            (df_processed['risk_score'] < 25) & (df_processed['anomaly'] == 'high_risk')
        ]
        
        if len(inconsistency) > 0:
            print(f"⚠️  Atenção: {len(inconsistency)} registros com inconsistência entre risk_score e anomaly")
        else:
            print("✓ Consistência entre risk_score e anomaly: OK")
            checks_passed += 1
    
    # 2. Padrões suspeitos - Transações muito grandes para novos usuários
    if all(col in df_processed.columns for col in ['amount', 'age_group']):
        total_checks += 1
        suspicious = df_processed[
            (df_processed['amount'] > 50000) & 
            (df_processed['age_group'] == 'new')
        ]
        
        if len(suspicious) > 0:
            print(f"⚠️  Possível anomalia: {len(suspicious)} transações grandes de usuários novos")
        else:
            print("✓ Padrão de transações por idade: OK")
            checks_passed += 1
    
    # 3. Verificar se todas as transações têm valores positivos
    if 'amount' in df_processed.columns:
        total_checks += 1
        negative_amounts = (df_processed['amount'] < 0).sum()
        if negative_amounts > 0:
            print(f"⚠️  Atenção: {negative_amounts} transações com valores negativos")
        else:
            print("✓ Valores de transações: OK")
            checks_passed += 1
    
    # 4. Verificar sessões com duração muito longa
    if 'session_duration' in df_processed.columns:
        total_checks += 1
        long_sessions = (df_processed['session_duration'] > 500).sum()
        if long_sessions > 0:
            print(f"⚠️  Atenção: {long_sessions} sessões com duração > 500 minutos")
        else:
            print("✓ Duração de sessões: OK")
            checks_passed += 1
    
    print(f"\nVerificações passadas: {checks_passed}/{total_checks}")

def run_enhanced_data_pipeline(file_path: str):
    """
    Pipeline de dados com validações adicionais
    """
    try:
        # 1. Importação 
        print("=== PIPELINE DE DADOS ===")
        print(f"Carregando arquivo: {file_path}")
        
        df = pd.read_csv(file_path)
        total_records = len(df)
        print(f"✓ Quantidade total de registros: {total_records:,}")
        
        # 2. Análise Exploratória Inicial 
        detailed_data_analysis(df)
        
        # 3. Validação de Qualidade 
        validation_issues = validate_data_quality(df)
        
        # 4. Limpeza e Tratamento 
        df_cleaned = enhanced_data_cleaning(df)
        
        # 5. Métricas Finais de Qualidade 
        print("\n=== MÉTRICAS FINAIS DE QUALIDADE ===")
        final_records = len(df_cleaned)
        data_quality_score = (final_records / total_records) * 100
        
        print(f"Registros iniciais: {total_records:,}")
        print(f"Registros após limpeza: {final_records:,}")
        print(f"Taxa de retenção: {data_quality_score:.2f}%")
        print(f"Registros removidos: {total_records - final_records:,}")
        
        # 6. Verificações de Consistência 
        post_processing_checks(df_cleaned)
        
        # 7. Análise Pós-Limpeza 
        print("\n=== RESUMO PÓS-LIMPEZA ===")
        print("Distribuição das variáveis principais:")
        if 'location_region' in df_cleaned.columns:
            print("\nRegiões (após limpeza):")
            print(df_cleaned['location_region'].value_counts())
        
        if 'transaction_type' in df_cleaned.columns:
            print("\nTipos de transação:")
            print(df_cleaned['transaction_type'].value_counts())
        
        if 'anomaly' in df_cleaned.columns:
            print("\nClassificação de risco:")
            print(df_cleaned['anomaly'].value_counts())
        
        # 8. Geração dos Relatórios 
        print("\n" + "="*60)
        print("RELATÓRIOS ANALÍTICOS")
        print("="*60)
        
        # Tabela 1: Média de Risk Score por Região
        if 'location_region' in df_cleaned.columns and 'risk_score' in df_cleaned.columns:
            tabela1 = df_cleaned.groupby('location_region')['risk_score'].agg(['mean', 'std', 'count']).round(2)
            tabela1 = tabela1.sort_values('mean', ascending=False)
            tabela1.rename(columns={'mean': 'average_risk_score'}, inplace=True)
            
            print("\n### Tabela 1: Estatísticas de Risk Score por Região ###")
            print(tabela1)
        else:
            print("⚠️  Dados insuficientes para gerar Tabela 1")
            tabela1 = None
        
        # Tabela 2: Top 3 Endereços por Valor em Transações 'Sale'
        if all(col in df_cleaned.columns for col in ['transaction_type', 'receiving_address', 'amount', 'timestamp']):
            sales_df = df_cleaned[df_cleaned['transaction_type'] == 'sale'].copy()
            
            if len(sales_df) > 0:
                sales_df = sales_df.sort_values('timestamp', ascending=True)
                latest_sales = sales_df.drop_duplicates(subset='receiving_address', keep='last')
                
                # Verificar se há dados suficientes
                if len(latest_sales) >= 3:
                    tabela2 = latest_sales.nlargest(3, 'amount')[['receiving_address', 'amount', 'timestamp']]
                else:
                    tabela2 = latest_sales[['receiving_address', 'amount', 'timestamp']]
                    print(f"⚠️  Atenção: Apenas {len(latest_sales)} transações 'sale' únicas encontradas")
            else:
                tabela2 = pd.DataFrame(columns=['receiving_address', 'amount', 'timestamp'])
                print("⚠️  Atenção: Nenhuma transação do tipo 'sale' encontrada")
            
            print("\n### Tabela 2: Top 3 Endereços por Valor em Transações 'Sale' ###")
            print(tabela2.reset_index(drop=True))
        else:
            print("⚠️  Dados insuficientes para gerar Tabela 2")
            tabela2 = None
        
        # --- 9. Relatório Final Consolidado ---
        print("\n" + "="*60)
        print("RELATÓRIO FINAL DO PIPELINE")
        print("="*60)
        print(f"✓ Dados processados com sucesso: {final_records:,} registros")
        print(f"✓ Qualidade geral dos dados: {data_quality_score:.2f}%")
        
        if tabela1 is not None:
            print(f"✓ Regiões analisadas: {len(tabela1)}")
        
        if tabela2 is not None:
            sales_count = len(df_cleaned[df_cleaned['transaction_type'] == 'sale']) if 'transaction_type' in df_cleaned.columns else 0
            print(f"✓ Transações 'sale' únicas: {len(latest_sales) if 'latest_sales' in locals() else sales_count}")
        
        if 'timestamp' in df_cleaned.columns:
            most_recent = df_cleaned['timestamp'].max()
            if pd.notnull(most_recent):
                print(f"✓ Transação mais recente: {most_recent.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print("⚠️  Não foi possível determinar a transação mais recente (timestamps ausentes ou inválidos)")
        else:
            print("⚠️  Coluna 'timestamp' não encontrada para determinar a transação mais recente")

        print(f"✓ Timestamp da execução: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        return df_cleaned, tabela1, tabela2
        
    except FileNotFoundError:
        print(f"❌ Erro: O arquivo no caminho '{file_path}' não foi encontrado.")
        return None, None, None
    except Exception as e:
        print(f"❌ Ocorreu um erro inesperado durante a execução do pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def save_df_clean(df_cleaned):
    """Função para salvar o DataFrame limpo em um arquivo CSV"""
    if df_cleaned is not None:
        output_path = 'data/df_fraud_credit_cleaned.csv'
        df_cleaned.to_csv(output_path, index=False)
        print(f"\n✓ DataFrame limpo salvo em: {output_path}")
    else:
        print("\n❌ Falha ao salvar o DataFrame limpo devido a erros no pipeline.")

# Executar o pipeline
if __name__ == '__main__':
    print("Iniciando pipeline de dados para análise de fraudes...")
    df_processed, tabela1, tabela2 = run_enhanced_data_pipeline('data/df_fraud_credit.csv')
    save_df_clean(df_processed)
    if df_processed is not None:
        print("\nPipeline executado com sucesso!")
    else:
        print("\nFalha na execução do pipeline.")
