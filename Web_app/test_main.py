from streamlit.testing.v1 import AppTest


def test_title():
    at = AppTest.from_file("main.py", default_timeout=60).run()
    assert "Welcome" in at.title[0].value
    assert "Welcome to the automated tumour detection service." in at.markdown[0].value


def test_selectbox_img():
    at = AppTest.from_file("main.py", default_timeout=30).run()
    assert at.selectbox[0].value == None
    assert at.selectbox[0].index == None
    at.selectbox[0].set_value("dcm").run()
    assert at.selectbox[0].value == "dcm"
    assert at.selectbox[0].index == 0
    at.selectbox[0].set_value("png").run()
    assert at.selectbox[0].value == "png"
    assert at.selectbox[0].index == 1
    at.selectbox[0].set_value("jpg").run()
    assert at.selectbox[0].value == "jpg"
    assert at.selectbox[0].index == 2
    at.selectbox[0].set_value("jpeg").run()
    assert at.selectbox[0].value == "jpeg"
    assert at.selectbox[0].index == 3


