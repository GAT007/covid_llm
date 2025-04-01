import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def home():
    st.title('Home Page')
    # Add content for home page

def matplotlib_plot():
    st.title('Matplotlib Plot Page')
    # Add content for Matplotlib plot page
    # In this example, we'll display a simple sin plot
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y)
    st.pyplot(plt)


def main():
    pages = {
        "Home": home,
        "Matplotlib Plot": matplotlib_plot
    }

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    page = pages[selection]
    page()

if __name__ == "__main__":
    main()

