import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def fusionner_csv(dossier):
    """
    Fusionne tous les CSV
    """
    # Liste pour stocker les DataFrames
    dataframes = []

    # Parcours tous les fichiers dans le dossier
    for fichier in os.listdir(dossier):
        # Vérifie si le fichier est un CSV
        if fichier.endswith('.csv'):
            # Chemin complet du fichier
            chemin_complet = os.path.join(dossier, fichier)
            # Lecture du CSV dans une DataFrame
            df = pd.read_csv(chemin_complet, delimiter=',')
            # Ajout de la DataFrame à la liste
            dataframes.append(df)

    # Fusionne toutes les DataFrames dans une seule
    df_final = pd.concat(dataframes, ignore_index=True)

    return df_final

def convertir_en_format_numerique(valeur):
    return float(format(float(valeur), '.2f'))


def remplacer_nan_par_moyenne_mensuelle(df):
    """
    Remplace les valeurs NaN dans la colonne 'market_cap' par la moyenne mensuelle correspondante.
    
    Paramètres :
    df (pandas.DataFrame) : DataFrame contenant les colonnes 'timestamp' et 'market_cap'.
    
    Retour :
    pandas.DataFrame : DataFrame avec les valeurs NaN remplacées par la moyenne mensuelle.
    """
    if 'timestamp' not in df.columns or 'market_cap' not in df.columns:
        raise ValueError("La DataFrame doit contenir les colonnes 'timestamp' et 'market_cap'.")

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Grouper par mois et calculer la moyenne mensuelle de 'market_cap'
    df['month'] = df['timestamp'].dt.to_period('M')
    monthly_mean = df.groupby('month')['market_cap'].transform('mean')

    # Remplacer les NaN par la moyenne mensuelle
    df['market_cap'] = df['market_cap'].fillna(monthly_mean)

    # Supprimer la colonne temporaire 'month'
    df = df.drop(columns=['month'])

    return df

def somme_marketcap_par_timestamp(df):
    """
    Agrège la somme de la capitalisation boursière (marketCap) par timestamp.
    
    Paramètres :
    df (pandas.DataFrame) : DataFrame contenant les colonnes 'timestamp' et 'market_cap'.
    
    Retour :
    pandas.DataFrame : DataFrame avec la somme de la capitalisation boursière par timestamp.
    """
    # Vérifie si les colonnes nécessaires sont présentes dans la DataFrame
    if 'snapped_at' not in df.columns or 'market_cap' not in df.columns:
        raise ValueError("La DataFrame doit contenir les colonnes 'snapped_at' et 'market_cap'.")
    df['timestamp'] = df['snapped_at'].apply(lambda x: x.split('T')[0])
    # Convertir 'timestamp' en type datetime pour une manipulation plus facile
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    print(df['timestamp'].unique())
    df = df.sort_values('timestamp')
    df_agg = df.groupby('timestamp')['market_cap'].sum().reset_index()
    df_agg['market_cap'] = df_agg['market_cap'].apply(convertir_en_format_numerique)
    # Grouper par mois et calculer la moyenne de 'market_cap'
    df_agg = df.groupby(pd.Grouper(key='timestamp', freq='D'))['market_cap'].max().reset_index()

    return df_agg

def plot_marketcap(df_marketcap, df_prices=None, btc=None, list_token=None):
    """
    Trace la capitalisation boursière au fil du temps avec un graphique en escalier,
    et optionnellement les prix sur un second axe y.
    
    Paramètres :
    df_marketcap (pandas.DataFrame) : DataFrame contenant les colonnes 'timestamp' et 'market_cap'.
    df_prices (pandas.DataFrame) : DataFrame optionnel contenant les colonnes 'Date', 'Open', 'High', 'Low', 'Close'.
    """
    # Préparation du DataFrame de capitalisation
    df_marketcap['timestamp'] = pd.to_datetime(df_marketcap['timestamp'])
    df_marketcap = df_marketcap.sort_values('timestamp')

    # Création du graphique
    _, ax1 = plt.subplots(figsize=(20, 8))

    # Tracé de la capitalisation boursière
    ax1.step(df_marketcap['timestamp'], df_marketcap['market_cap'], where='post', label='Market Cap (Step)', color='green')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Market Cap', color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=15))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xlim([df_marketcap['timestamp'].min(), pd.Timestamp('2024-05-22')])

    # Si df_prices est fourni, tracer les prix sur un second axe y
    if df_prices is not None:
        ax2 = ax1.twinx()  # Créer un deuxième axe y qui partage le même axe x
        ax2.plot(pd.to_datetime(df_prices['Date']), df_prices['Close'], label='NVIDIA Price', color='red')
        ax2.set_ylabel('Price', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

    if btc is not None:
        ax3 = ax1.twinx()  # Créer un deuxième axe y qui partage le même axe x
        ax3.plot(pd.to_datetime(btc['Date']), btc['Close'], label='BTC Price', color='orange')
        ax3.set_ylabel('Price')
        ax3.tick_params(axis='y', labelcolor='orange')

        # Ajout d'une légende pour les deux axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax3.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper left')

    if list_token is not None :
        plt.annotate(list_token, xy=(0.5, 0), xycoords='axes fraction', ha='center', va='top', fontsize=10, xytext=(0, -50), textcoords='offset points')

    plt.title('Market Cap and Close Price Over Time')
    plt.show()
