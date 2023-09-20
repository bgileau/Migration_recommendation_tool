DESCRIPTION
Moving to a new location is a potentially life-changing and complex decision which upends all
aspects of one's life. A fundamental lack of resources and tools unfortunately exists when
determining where to move. Despite the ongoing decline in migration rates over the past few decades,
approximately 25 million, or 8\% of Americans, still migrate within the United States each year,
with many more considering relocation. With the post-COVID rise of remote work along with other
factors such as increased gig-economy employment, job location and residence choice have become
decoupled for many Americans. Accordingly, financial affordability along with other features (e.g.
weather or political environment) are increasingly driving relocation decisions. Since no such tool
exists to serve this ever-increasing gap in demand, we've aggregated several datasets and have
chosen many features relevant to us personally and identified in the literature survey to develop a
tool that serves the needs of those considering relocation, as well as government officials seeking
to increase migration to their own area\cite{a1}.

There are no existing tools that compare and visualize counties in the U.S. with a focus on
county-to-county migration, taking into account features custom-selected by the user, such as
financial affordability, weather, and political environment.

We developed a visually engaging and interactive tool that allows users to understand how their
feature preferences vary across counties. Through selection of features and values, users can
receive personalized and data-driven recommendations.


INSTALLATION
Assumptions:
    Python 3.10 installed (other versions might work)
    Python venv installed
        Windows and MacOS should already have it
        For Linux run: sudo apt install python3.10-venv
    Project unzipped

Run the following steps in your system's shell:
    1. Change to CODE directory:
        (All)       cd <project path>/CODE

    2. Create a virtual environment:
        (Win)       py -3.10 -m venv venv
        (Mac/Linux) python3.10 -m venv venv

    3. Activate the virtual environment:
        (Win)       ./venv/Scripts/Activate.ps1
        (Mac/Linux) source venv/bin/activate

    4. Install the requirements:
        (All)       pip install -r requirements.txt

    5. Run the application:
        (All)       streamlit run main.py
        Note: You can just press enter if streamlit asks for an email

    6. Stop the application:
        Press Ctrl + c in the shell
        Note: On Windows, you need to have the app webpage still open for Ctrl + c to work

    7. Deactivate virtual environment:
        (All)       deactivate


EXECUTION
Once you've run the application using the above steps, a webpage should open, or you can go to:
http://localhost:8501

1.  Start by entering a ZIP code (and county if needed) and an income on the left.
2.  Choose features from the dropdown in order of their importance to you.
    Note: Home Value, Population, and Political Climate let you choose your ideal value.
3.  Press "Tell me where to move!"
4.  You'll see a map in the main view showing the overall score for all counties. You can drag
    and zoom the map, and the current and recommended counties are highlighted. Mouseover a
    county to see its score.
5.  Below the map are the top recommended places. Click on any of them to see more information.
6.  Use the buttons below the recommendations to see more or fewer results.
7.  Click "Show Supporting Data" to see the data that did or could have gone into the score.
8.  You can change the feature shown with the dropdown and explore the map as above. The
    histogram below the map shows the distribution of all counties for that feature.
