from utils import *
import pickle
import streamlit as st
from gensim.models import Word2Vec, KeyedVectors
from tensorflow import keras

st.set_page_config(page_title="English to French Translator")
st.markdown('# English to French Translation Tool')
st.markdown('<p style="color:grey; font-size: 18px;">By Sai Shreyas Bhavanasi</p>',
            unsafe_allow_html=True)

if 'count' not in st.session_state:
    st.session_state['count'] = ""
if 'key' not in st.session_state:
    st.session_state.key = 'value'
if 'clicks' not in st.session_state:
    st.session_state['clicks'] = 0


@st.cache(allow_output_mutation=True)
def get_xmod():
    x_mod = KeyedVectors.load_word2vec_format(
        'eng_20k.vecs', binary=False)
    print()
    return x_mod


@st.cache(allow_output_mutation=True)
def get_ymod():
    y_mod = KeyedVectors.load_word2vec_format(
        'fr_20k.vecs', binary=False)
    print('here')

    return y_mod


@st.cache(allow_output_mutation=True)
def get_model():
    model = keras.models.load_model("en-fr.keras")
    print()

    return model


@st.cache(allow_output_mutation=True)
def get_tables():
    with open('tables.pkl', 'rb') as f:
        tables = pickle.load(f)
    ins_table = tables['ins']
    del_table = tables['del']
    sub_table = tables['sub']
    return ins_table, del_table, sub_table


def get_nn_preds(y_mod, mod_res):
    predictions = y_mod.most_similar(mod_res, topn=10)
    assert len(predictions) == 10
    preds, dist = zip(*predictions)
    return preds


x_mod = get_xmod()
model = get_model()
y_mod = get_ymod()
if 'ins_table' not in st.session_state:
    st.session_state['ins_table'], st.session_state['del_table'], st.session_state['sub_table'] = get_tables()


def get_translation():

    st.session_state.clicks += 1

    input_word = st.session_state['inps']
    input_word = input_word.lower()
    try:
        input_vec = x_mod[input_word].reshape(1, -1)
    except KeyError:
        st.session_state.count = st.session_state['inps'] + \
            " not in model vocabulary"
        return
    out_vec = model.predict(input_vec)
    preds = get_enhanced_preds(
        y_mod, out_vec, input_word,  st.session_state['ins_table'], st.session_state['del_table'], st.session_state['sub_table'])
    st.session_state.count = preds


st.text_input('Enter English Word', on_change=get_translation,
              key='inps', value="dog")
st.button('Get Translation', on_click=get_translation)


if st.session_state.clicks > 0:
    st.write("Top 10 French Translations:")
    if type(st.session_state.count) is list:
        s = ''
        for pred in st.session_state.count:
            s += "- " + pred + "\n"
        st.markdown(s)
    else:
        st.write(st.session_state.inps + " not in vocabulary")

st.write("""
         ---
         The translation model is a replication of the paper [Exploiting Similarities among Languages for Machine Translation](https://arxiv.org/pdf/1309.4168.pdf). 
         This model is unsupervised i.e. translates with minimal parallel data (5000 examples). Traditonal supervised translations require large 
         corpuses of parallel translated data which doesn't scale well. This model has applications such as language translation and extending dictionaries.
         
         Further, some french words have similar translations to their english counterparts (cat -> chat/chatte). To exploit this property, I refine the predictions
         by ranking the top 400 predictions based on the edit distance (spelling similarity).
         """)
