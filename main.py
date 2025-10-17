import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
import numpy as np
import os # Added for file operations

# --- 1. Page Configuration and Constants ---
st.set_page_config(
    page_title="Online Shopping Behavior Analysis: Age vs. Purchase Amount",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page name constants
PAGE_HOME = "Project Overview"
PAGE_HYPOTHESIS = "Hypothesis Test: 'Age' vs. 'Purchase Amount'"
PAGE_FACTORS = "Key Purchase Drivers"
PAGE_SEGMENTATION = "Customer Pattern Analysis: Age-Based Customer Behavior"
PAGE_RESOURCES = "Download Resources" # New page constant

# Chart style constant
CHART_TEMPLATE = "plotly_white"

# --- 2. Data Loading and Preprocessing ---
try:
    # Read the CSV file
    df = pd.read_csv('Ecommerce_Consumer_Behavior_Analysis_Data.csv')
    
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    
    # Convert the purchase_amount column from string to number
    # Remove '$' symbol, strip whitespace, and cast to float
    df['purchase_amount'] = df['purchase_amount'].str.replace('$', '', regex=False).str.strip().astype(float)

    # Drop rows with missing data
    df.dropna(inplace=True)
    
except FileNotFoundError:
    st.error("'Ecommerce_Consumer_Behavior_Analysis_Data.csv' not found. Please ensure the file is in the same directory as main.py.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during data processing: {e}")
    st.stop()


# --- 3. Page Rendering Functions ---

def render_home_page(df: pd.DataFrame):
    """Renders the 'Project Overview' page."""
    st.title("🛒 Hypothesis Test: Is Age a Key Driver in Online Shopping?")
    st.markdown("---")
    st.header("Project Objective")
    st.write("This project began with the common assumption that **'older adults shop online less than younger adults.'** The objective is to validate this hypothesis through data analysis and, further, to identify the key factors that actually influence online purchasing behavior.")
    st.header("Executive Summary")
    st.write("The analysis reveals **no statistically significant relationship between age and purchase amount.** This finding does not support the initial hypothesis. Instead, behavioral variables, such as the **'product category,'** demonstrate a more substantial impact on purchase amount. Furthermore, while the average purchase amount is similar across age groups, there are qualitative differences in spending patterns, such as **'infrequent, high-value purchases'** versus **'frequent, low-value purchases.'**")
    st.markdown("---")
    st.header("Data Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gender Distribution")
        gender_counts = df['gender'].value_counts()
        fig = px.pie(gender_counts, values=gender_counts.values, names=gender_counts.index, title="Customer Gender Distribution", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(template=CHART_TEMPLATE, title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Age Distribution")
        fig = px.histogram(df, x='age', nbins=30, title="Customer Age Distribution", marginal="box")
        fig.update_layout(template=CHART_TEMPLATE, title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)

def render_hypothesis_page(df: pd.DataFrame):
    """Renders the 'Hypothesis Test' page."""
    st.header("📊 Hypothesis 1: Analyzing the Relationship Between Age and Purchase Amount")
    st.write("This section provides an in-depth validation of the core hypothesis—the relationship between age and purchase amount—using both visual and statistical methods.")
    st.markdown("---")
    st.subheader("1. Visual Analysis: Scatter Plot and Trendline")
    fig_scatter = px.scatter(df, x='age', y='purchase_amount', title="Relationship Between Age and Purchase Amount (Scatter Plot)", trendline="ols", trendline_color_override="red", labels={'age': 'Age', 'purchase_amount': 'Purchase Amount (USD)'})
    fig_scatter.update_layout(template=CHART_TEMPLATE, title_x=0.5)
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.info("The wide distribution of data points and the **nearly horizontal OLS trendline** visually suggest the absence of a distinct linear relationship between age and purchase amount.")
    st.markdown("---")
    st.subheader("2. Statistical Analysis: Comparing Mean Purchase Amounts Between Two Age Groups (T-test)")
    st.write("Here, we test whether there is a statistically significant difference in the mean purchase amount between two user-defined age groups.")
    st.sidebar.subheader("Define Analysis Groups")
    min_age, max_age = int(df['age'].min()), int(df['age'].max())
    group1_ages = st.sidebar.slider('🔵 Group 1 (e.g., Younger Adults) Age Range', min_age, max_age, (20, 35))
    group2_ages = st.sidebar.slider('🔴 Group 2 (e.g., Older Adults) Age Range', min_age, max_age, (45, 60))
    group1 = df[(df['age'] >= group1_ages[0]) & (df['age'] <= group1_ages[1])]['purchase_amount']
    group2 = df[(df['age'] >= group2_ages[0]) & (df['age'] <= group2_ages[1])]['purchase_amount']
    if len(group1) < 2 or len(group2) < 2:
        st.warning("Insufficient data for the selected age ranges. Please adjust the sliders to perform the analysis.")
        return
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label=f"Group 1 Mean Purchase ({group1_ages[0]}-{group1_ages[1]} yrs)", value=f"${group1.mean():.2f}")
        st.metric(label=f"Group 2 Mean Purchase ({group2_ages[0]}-{group2_ages[1]} yrs)", value=f"${group2.mean():.2f}")
        st.metric(label="p-value", value=f"{p_value:.4f}")
    with col2:
        if p_value < 0.05:
            st.error(f"**Conclusion (p < 0.05):** There is a **statistically significant difference** in the mean purchase amount between the two groups.")
        else:
            st.success(f"**Conclusion (p >= 0.05):** There is **no statistically significant difference** in the mean purchase amount between the two groups.")
        st.write("The p-value represents the probability that the observed difference in means occurred by chance. A p-value less than 0.05 is typically considered statistically significant.")
    st.markdown("---")
    st.subheader("3. Predictive Model Analysis: Simple Linear Regression")
    st.markdown("We use a `Purchase Amount ~ Age` model to determine how well age explains the variance in purchase amount.")
    X = sm.add_constant(df['age'])
    y = df['purchase_amount']
    model = sm.OLS(y, X).fit()
    r_squared = model.rsquared
    p_value_age = model.pvalues['age']
    st.write(f"**Model Explanatory Power (R-squared):** `{r_squared:.4f}`")
    st.write(f"**P-value for the Age Variable:** `{p_value_age:.4f}`")
    st.info(f"**Analysis:**\n- The **R-squared value** is very close to 0, indicating that this model explains almost none of the variability in purchase amount.\n- The **p-value for the age variable** is much larger than 0.05, confirming that age does not have a statistically significant effect on purchase amount.\n- **In conclusion, the statistical model clearly shows that 'age' alone is not a reliable predictor of purchase amount.**")

def render_factors_page(df: pd.DataFrame):
    """Renders the 'Deeper Analysis' page."""
    st.header("💡 Deeper Analysis: The Real Factors Influencing Purchase Amount")
    st.write("If 'age' isn't a key driver, which variables have a significant impact on purchase amount? We explore this using multiple linear regression.")
    st.markdown("`Purchase Amount ~ Age + Category + Purchase Frequency + Gender`")
    st.markdown("---")
    df_model = pd.get_dummies(df.copy(), columns=['purchase_category', 'frequency_of_purchase', 'gender'], drop_first=True, dtype=int)
    X_cols = ['age'] + [col for col in df_model.columns if col.startswith(('purchase_category_', 'frequency_of_purchase_', 'gender_'))]
    X_multi = sm.add_constant(df_model[X_cols])
    y = df_model['purchase_amount']
    model_multi = sm.OLS(y, X_multi).fit()
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Multiple Regression Results Summary")
        st.text(model_multi.summary())
    with col2:
        st.subheader("Visualizing Variable Influence (Regression Coefficients)")
        coeffs = pd.DataFrame({'coefficient': model_multi.params, 'p_value': model_multi.pvalues}).drop('const')
        coeffs['significant'] = coeffs['p_value'] < 0.05
        coeffs = coeffs.sort_values('coefficient', ascending=False)
        fig = px.bar(coeffs, x='coefficient', y=coeffs.index, orientation='h', color='significant', color_discrete_map={True: 'salmon', False: 'lightgrey'}, labels={'coefficient': 'Coefficient (Impact on Purchase Amount)', 'y': 'Variable'}, title="Impact of Each Variable on Purchase Amount")
        fig.update_layout(template=CHART_TEMPLATE, title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)
    st.info("**Analysis:**\n- **Age:** The p-value remains high, indicating it is not statistically significant (gray bar).\n- **Purchase Category:** Certain categories have a significant positive (+) or negative (-) impact on the purchase amount (salmon-colored bars).\n- **Conclusion: This reveals that *what* a customer buys (category) is a far more important predictor of purchase amount than their age.**")

def render_segmentation_page(df: pd.DataFrame):
    """Renders the 'Customer Pattern Analysis' page."""
    st.header("🧩 Customer Pattern Analysis: Consumption Behavior by Age Group")
    st.write("Average purchase amount alone is insufficient for understanding customers. We segment customers into four groups based on 'purchase value' and 'purchase frequency' and analyze the distribution across age groups to identify differences in consumption 'patterns'.")
    st.markdown("---")
    median_purchase = df['purchase_amount'].median()
    high_freq_list = df['frequency_of_purchase'].value_counts().nlargest(3).index.tolist()
    df_seg = df.copy()
    df_seg['value_segment'] = np.where(df_seg['purchase_amount'] > median_purchase, 'High-Value', 'Low-Value')
    df_seg['freq_segment'] = np.where(df_seg['frequency_of_purchase'].isin(high_freq_list), 'High-Frequency', 'Low-Frequency')
    conditions = [(df_seg['freq_segment'] == 'High-Frequency') & (df_seg['value_segment'] == 'High-Value'), (df_seg['freq_segment'] == 'Low-Frequency') & (df_seg['value_segment'] == 'High-Value'), (df_seg['freq_segment'] == 'High-Frequency') & (df_seg['value_segment'] == 'Low-Value')]
    choices = ['Champion', 'Big Spender', 'Loyal Customer']
    df_seg['customer_segment'] = np.select(conditions, choices, default='Potential Customer')
    age_conditions = [df_seg['age'].between(20, 35), df_seg['age'].between(45, 60)]
    age_choices = ['Younger Adults (20-35)', 'Older Adults (45-60)']
    df_seg['age_group'] = np.select(age_conditions, age_choices, default='Other')
    seg_dist = df_seg[df_seg['age_group'] != 'Other'].groupby(['age_group', 'customer_segment']).size().reset_index(name='count')
    st.subheader("Customer Segment Distribution by Age Group")
    fig = px.bar(seg_dist, x='age_group', y='count', color='customer_segment', title='Customer Segment Distribution by Age Group', labels={'count': 'Customer Count', 'age_group': 'Age Group', 'customer_segment': 'Customer Segment'}, barmode='group', template=CHART_TEMPLATE, color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)
    st.info("**Analysis:**\n- This chart illustrates the differences in 'consumption patterns' between the two age groups. For example, younger adults might have a higher proportion of 'Loyal Customers' (frequent, low-value purchases), while older adults may have a higher concentration of 'Big Spenders' (infrequent, high-value purchases).\n- **Conclusion: Even if the average purchase amounts are similar, the *way* different age groups spend can vary significantly. This suggests that distinct marketing strategies may be required for each age demographic.**")

# MODIFIED function to render the resources page
def render_resources_page(df: pd.DataFrame):
    """Renders the 'Download Resources' page with a new layout."""
    st.header("📂 Downloadable Resources")
    st.write("Access all project-related files and deliverables from Group15.")
    st.markdown("---")
    
    try:
        script_name = os.path.basename(__file__)
    except NameError:
        script_name = "main.py" 

    files_in_directory = sorted(os.listdir('.'))
    files_to_exclude = [script_name, 'Ecommerce_Consumer_Behavior_Analysis_Data.csv']

    for file_name in files_in_directory:
        if os.path.isfile(file_name) and file_name not in files_to_exclude:
            col1, col2 = st.columns([3, 1])  # 3:1 ratio for text and button
            
            with col1:
                # Add some vertical space to better align with the button
                st.markdown(f"<div style='padding-top: 8px;'>📄 Check Group15's <b>{file_name}</b>!</div>", unsafe_allow_html=True)

            with col2:
                try:
                    with open(file_name, "rb") as file_data:
                        st.download_button(
                            label="Download",
                            data=file_data,
                            file_name=file_name,
                            mime='application/octet-stream',
                            key=f"download_btn_{file_name}" # Unique key is crucial
                        )
                except Exception as e:
                    st.warning(f"Could not read {file_name}.")
            
            st.markdown("---") # Separator for the next file entry

# --- 4. Main App Execution Logic ---
def main():
    """Main function to run the Streamlit app"""
    st.sidebar.title("Analysis Menu")
    page = st.sidebar.radio(
        "Select a page",
        [PAGE_HOME, PAGE_HYPOTHESIS, PAGE_FACTORS, PAGE_SEGMENTATION, PAGE_RESOURCES] # Added new page
    )

    page_functions = {
        PAGE_HOME: render_home_page,
        PAGE_HYPOTHESIS: render_hypothesis_page,
        PAGE_FACTORS: render_factors_page,
        PAGE_SEGMENTATION: render_segmentation_page,
        PAGE_RESOURCES: render_resources_page # Added new page function
    }
    
    page_functions[page](df)

if __name__ == "__main__":
    main()