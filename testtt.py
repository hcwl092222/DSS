import streamlit as st
import os



# 更改工作目錄
current = os.path.dirname(os.path.abspath(__file__))
os.chdir(current)

names=['angela']
username=["leemaki138"]
password=["1234567"]

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    if st.button("Log in"):
        st.session_state.logged_in = True
        st.rerun()

def logout():
    if st.button("Log out"):
        st.session_state.logged_in = False
        st.rerun()
        
def main():
# 檢查使用者是否登入
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        login()  # 如果尚未登入，顯示登入表單
        return  # 讓用戶登入後才能繼續
    logout()

if __name__=='_main_':
    main()

login_page = st.Page(login, title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")


dashboard = st.Page(
    "streamlit_ex.py", title="Dashboard", icon=":material/dashboard:", default=True
)

bugs = st.Page("test2.py", title="Prediction", icon=":material/dashboard:")
if st.session_state.logged_in:
    pg = st.navigation(
        {
            "Account": [logout_page],
            "Reports": [dashboard , bugs],
        }
    )
else:
    pg = st.navigation([login_page])

pg.run()