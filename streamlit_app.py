import calendar
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from streamlit.logger import get_logger
from pyplants.core.base import BaseDisease
from pyplants.core.context import UpdateCtx
from pyplants.phenology.iphen import Iphen
from pyplants.utils.plants import PlantEnum
from pyplants.utils.helpers import DailyUpdater
from pyplants.diseases.grape.pm import Gadoury, Moyer, DavisRI, Caffi
from pyplants.diseases.grape.gm import Broome, GoFe, GoDom
from pyplants.diseases.grape.dm import Plasmo

# Supported grape diseases
DISEASES = {
    "Powdery Mildew": "spore_release",
    "Grey Mould": "infection",
    "Downy Mildew": "infection"
}

# Cultivar supported for Phenology
CULTIVAR = [
    "CABERNET",
    "ALBANA",
    "BARBERA",
    "BELLONE",
    "BERVEDINO",
    "BIANCHETTA",
    "BOMBINO",
    "BONAMICO",
    "BRACHETTO",
    "CAGNULARI",
    "CATANESE",
    "CATARRATTO",
    "CHARDONNAY",
    "CILIEGIOLO",
    "CORTESE",
    "CORVINA",
    "DOLCETTO",
    "DURELLA",
    "EMPIBOTTE",
    "ERBALUCE",
    "FIANO",
    "FREISA",
    "GAGLIOPPO",
    "GARGANEGA",
    "GRECHETTO",
    "GRIGNOLINO",
    "GRILLO",
    "LAGREIN",
    "LAMBRUSCO",
    "MALBEC",
    "MALVASIA",
    "MAMMOLO",
    "MARZEMINO",
    "MERLOT",
    "MOLINARA",
    "MONTEPULCIANO",
    "MOSCATO",
    "MULLER",
    "NASCO",
    "NEBBIOLO",
    "NEGRARA",
    "OTTAVIANELLO",
    "PAVANA",
    "PICCOLA",
    "PIGNOLA",
    "PINELLA",
    "PINOT",
    "RABOSO",
    "RIESLING",
    "ROSSARA",
    "SANGIOVESE",
    "SCHIAVA",
    "SCIASCINOSO",
    "SOMARELLO",
    "SUSUMANIELLO",
    "TEROLDEGO",
    "TINTORA",
    "TOCAI",
    "TRAMINER",
    "TREBBIANO",
    "TSCHAGGELE",
    "VERDELLO",
    "VERDUZZO",
    "VERMENTINO",
    "VESPOLINA"
]

HOME_PLACEHOLDER = """
## Welcome in PyPlants UI

This is a simple tool to execute models for grapevine included in `pyplants`. Please use the controls in the sidebar to load your data.

The csv file you upload must includes the following headers.

| COLUMN | DESCRIPTION | UNIT / FORMAT |
|---|---|---|
| DATE | Datetime |YYYY-mm-dd HH:MM:SS |
| TEMPERATURE | Air Temperature | °C |
| RH | Relative Humidity | % |
| LEAF_TOP | Leaf Wetness (Top side) | % |
| RAIN | Rainfalls | mm |

Please note that data must be hourly sampled and with no missing values.
"""

@st.cache_data
def load_data(uploaded_file, lw_threshold):
    """Load the dataset.

    :param uploaded_file: the csv file to use
    :param lw_threshold: the leaf wetness threshold to use
    :returns: the dataset in a pandas DataFrame
    """
    logger = get_logger(__name__)
    logger.info("Reading user provided data")
    # The expected columns inside the dataset
    expected_columns = ["DATE", "TEMPERATURE", "RH", "LEAF_TOP", "RAIN"]
    try:
        df = pd.read_csv(uploaded_file, parse_dates=["DATE"])
    except ValueError:
        df = None
        logger.warning("Unable to create the dataframe")
    else:
        # Check columns name inside the dataframe
        if not all(item in df.columns for item in expected_columns):
            df = None
            logger.warning("Missing columns in provided dataset")
        else:
            # Create the LEAF column using the threshold
            df["LEAF"] = (df["LEAF_TOP"] > lw_threshold).astype(int)
            # Set timezone and index of the dataframe
            df.set_index("DATE", inplace=True)
            # Localize with proper timezone
            df.index = df.index.tz_localize("UTC").tz_convert("Europe/Rome")
    return df


@st.cache_data
def run_models(df, cultiv: str, disease: str):
    """Run the pyplants models.

    :param df: the agrometeorological dataframe
    :param cultiv: the cultiv selected
    :param disease: the disease selected
    """
    logger = get_logger(__name__)
    logger.info("Running pyplants models!")
    models = []
    iphen = Iphen.build_from_plant(PlantEnum.GRAPE, cultiv.upper())
    # Select the models based on disease
    if disease == "Powdery Mildew":
        models.append(Gadoury(iphen))
        models.append(Moyer())
        models.append(DavisRI(iphen))
        models.append(Caffi(iphen))
    elif disease == "Downy Mildew":
        models.append(Plasmo(iphen))
    elif disease == "Grey Mould":
        models.append(Broome(iphen))
        models.append(GoFe(iphen))
        models.append(GoDom(iphen))
    else:
        raise ValueError("Unknown disease provided!")
    # Check daily model and register them in DailyUpdater
    daily_models = []
    hourly_models = []
    for model in models:
        if model.get_metadata()["timestep"] == "d":
            daily_models.append(model)
        else:
            hourly_models.append(model)
    daily_updater = DailyUpdater(daily_models)
    # Update all target models
    for row in df.itertuples():
        dt = row.Index.to_pydatetime()
        uctx = UpdateCtx(
            dt=dt,
            tmean=row.TEMPERATURE,
            tmax=row.TEMPERATURE,
            rhmean=row.RH,
            lw=row.LEAF,
            rain=row.RAIN)
        # Models update with new sample
        iphen.update(uctx)
        # Update hourly models
        for model in hourly_models:
            model.update(uctx)
        # Update daily models
        daily_updater.update(uctx)
    return iphen, models


def render_phenology(iphen: Iphen):
    """Render the phenology results.

    :param iphen: the used phenology model
    """
    v = [{"v": r.stage.code, "dt": r.dt} for r in iphen.scale.vscale]
    r = [{"r": r.stage.code, "dt": r.dt} for r in iphen.scale.rscale]
    v = pd.DataFrame(v)
    r = pd.DataFrame(r)
    st.subheader("Phenology", divider=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(v["dt"], v["v"], label='Vegetative Scale', color='limegreen')
    ax.plot(r["dt"], r["r"], label='Reproductive Scale', color='darkorange')
    ax.set_xlabel('Date')
    ax.set_ylabel('BBCH Stage')
    ax.set_yticks([0, 19, 39, 59, 79, 99])
    ax.legend(loc="lower right", fancybox=False)
    ax.grid(True, linestyle='--', color="#ddd")
    st.pyplot(fig)


def render_heatmap(df_input: pd.DataFrame, step: str, value: str):
    """Render a heatmap for the given value using a tabular calendar style.

    :param df_input: output of a disease model as pandas DataFrame
    :param step: the step of output (`h` or `d`)
    :param value: the column to use as value
    """
    df_input.set_index("dt", inplace=True)
    # Since plot is daily, we resample smaller timestep
    if step == "h":
        df_input = df_input.resample("D").max()
    # Remove timezone info to simplify computing
    df_input.index = df_input.index.tz_localize(None)
    # Determine target year
    year = df_input.index[0].year
    # Generate complete date range for the entire year (Jan 1 to Dec 31)
    full_year = pd.date_range(
        start=f'{year}-01-01',
        end=f'{year}-12-31',
        freq='D'
    )
    df_full = pd.DataFrame({'dt': full_year})
    # Merge input onto full year frame; missing dates take NaN values
    df = df_full.join(df_input, on="dt")
    # Extract Day of Week (0 = Monday, 6 = Sunday)
    df['day_of_week'] = df['dt'].dt.dayofweek
    # Compute week column index relative to the first week of the year
    df["week"] = df["dt"].dt.isocalendar().week - 1
    # Pivot to 7 x W matrix (Monday=0 to Sunday=6)
    pivot = df.pivot(
        index='day_of_week',
        columns='week',
        values=value
    ).reindex(range(7))
    # Build custom green (0.0) to red (1.0) color map
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'GreenToRed', ['#2ca02c', '#ff7f0e', '#d62728']
    )
    fig, ax = plt.subplots(figsize=(15, 3))
    # Set background to gray for unprovided data points
    ax.set_facecolor('#ccc')
    # Render heatmap with small white gaps between daily squares
    sns.heatmap(
        pivot,
        cmap=cmap,
        vmin=0,
        vmax=1,
        square=True,
        linewidths=1.2,
        linecolor='white',
        cbar_kws={
            'label': 'Risk',
            'orientation': 'horizontal',
            'pad': 0.25,
        },
        ax=ax,
    )
    # Y-axis formatting: Monday to Sunday
    days = list(calendar.day_abbr)
    ax.set_yticks([i + 0.5 for i in range(7)])
    ax.set_yticklabels(days, rotation=0)
    # X-axis formatting: Month locators positioned at the start of each month
    month_starts = pd.date_range(
        start=f'{year}-01-01',
        end=f'{year}-12-01',
        freq='MS'
    )
    month_ticks = [m.isocalendar().week - 1 for m in month_starts]
    month_names = [m.strftime('%b') for m in month_starts]
    ax.set_xticks([w + 0.5 for w in month_ticks])
    ax.set_xticklabels(month_names, rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(value.replace("_", " ").capitalize(), pad=12)
    plt.tight_layout()
    st.pyplot(fig)


def render_diseases(models: list[BaseDisease], disease: str):
    """Render an output table for each model.

    :param models: list of models executed
    :param disease: the disease name selected
    """
    for model in models:
        event_rows = []
        for event in model.events:
            event_rows.append(event.to_dict())
        # Use docstring to print sub headers
        model_name = model.__doc__.split("\n")[0]
        model_name = model_name.replace(".", "")
        st.subheader(model_name, divider=True)
        # Prepare dataframe
        df = pd.DataFrame(event_rows)
        step = model.get_metadata()["timestep"]
        # Plot the tabular output
        st.dataframe(df)
        render_heatmap(df, step, DISEASES[disease])


def validate_form(uploaded_file, cultiv, disease):
    """Validate form upon submission.

    :param uploaded_file: the file uploaded by the user
    :param cultiv: the selected cultivar
    :param disease: the selected disease
    :returns: a boolean indicating if the form submission is valid or not
    """
    has_error = False
    if uploaded_file is None:
        has_error = True
        st.error("Please provide a csv file.", icon=":material/error:")
    if cultiv is None:
        has_error = True
        st.error("Please provide a cultivar.", icon=":material/error:")
    if disease is None:
        has_error = True
        st.error("Please provide a disease.", icon=":material/error:")
    return not has_error

# Generic application configuration
st.set_page_config(page_title="PyPlants UI")

# Layout definition
with st.sidebar.form(key="options", border=False):
    st.title("PyPlants UI")
    st.subheader(":material/settings: Settings", divider=True)
    # File to analyze
    uploaded_file = st.file_uploader("Choose a file")
    # Leaf wetness threshold used to create LWD
    lw_threshold = st.number_input(
        "LEAF WETNESS THRESHOLD",
        min_value=0,
        max_value=100,
        value=40,
        step=1,
        help="Leaf wetness threshold to consider as wet a leaf")
    # Cultivar the measures belong
    cultiv = st.selectbox(
        "C.V.",
        index=None,
        options=CULTIVAR,
        help="Target cultivar")
    # Which disease we want to run the models
    disease = st.selectbox(
        "Disease",
        index=None,
        options=list(DISEASES.keys()),
        help="Target disease to analyze")
    # Submit button...
    ex_submitted = st.form_submit_button(
        ":material/play_circle: Run Models",
        type="primary")

placeholder = st.empty()
placeholder.markdown(HOME_PLACEHOLDER)

if ex_submitted:
   if validate_form(uploaded_file, cultiv, disease):
        placeholder.empty()
        with st.spinner("Now Loading..."):
            # Form is correctly submitted load dataframe
            df = load_data(uploaded_file, lw_threshold)
            if df is None:
                st.error(
                    "Provided file is malformed. Please check your input.",
                    icon=":material/error:")
                st.stop()
            # Execute plant models
            phen, models = run_models(df, cultiv, disease)
        # Plot phenology BBCH
        render_phenology(phen)
        # Plot disease results
        render_diseases(models, disease)
