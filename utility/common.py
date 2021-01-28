import logging

# ---------------------------------------------------------
# Global item management (figures, dataframes to save etc)

# A global list to keep figures and dataframes (to be saved at the end of the program)
figures_list = []
dataframes_list = []

def add_figure_to_save(fig, name=None):
    figures_list.append((fig, name))

def add_dataframe_to_save(df, name=None):
    dataframes_list.append((df, name))

def clear_all_figures():
    for (fig, name) in figures_list:
        fig.clear()
    figures_list.clear()

def clear_all_dataframes():
    dataframes_list.clear()

def save_all_figures(dir):
    for i, (fig, name) in enumerate(figures_list):
        figname = ''
        if name: figname = '_' + name

        filename = dir + '/fig_' + str(i+1) + figname + '.png'
        fig.savefig(filename)
    logging.info('All figures saved to {}'.format(dir))

def save_all_dataframes(dir):
    for i, (df, name) in enumerate(dataframes_list):
        dfname = ''
        if name: dfname = '_' + name

        filename = dir + '/df_' + str(i+1) + dfname + '.csv'
        df.to_csv(filename)
    logging.info('All dataframes saved to {}'.format(dir))