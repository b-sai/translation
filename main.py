import streamlit as st
from gensim.models import Word2Vec, KeyedVectors
from tensorflow import keras
st.title('English - French Translation Tool')
if 'count' not in st.session_state:
    st.session_state['count'] = ""
if 'key' not in st.session_state:
    st.session_state.key = 'value'


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
    print()

    return y_mod


@st.cache(allow_output_mutation=True)
def get_model():
    model = keras.models.load_model("en-fr.keras")
    print()

    return model


def get_nn_preds(model, mod_res):
    predictions = y_mod.most_similar(mod_res, topn=10)
    assert len(predictions) == 10
    preds, dist = zip(*predictions)
    return preds


x_mod = get_xmod()
model = get_model()
y_mod = get_ymod()


def get_translation():
    input_word = st.session_state['inps']
    input_word = input_word.lower()
    try:
        input_vec = x_mod[input_word].reshape(1, -1)
    except KeyError:
        st.session_state.count = st.session_state['inps'] + \
            " not in model vocabulary"
        return
    out_vec = model.predict(input_vec)
    preds = get_nn_preds(model, out_vec)
    # print(out_vec)
    st.session_state.count = preds


def demo(inp):
    st.write(st.session_state.inps + inp)


st.text_input('Enter English Word', on_change=get_translation,
              key='inps', value="hello")
st.button('Get Translation', on_click=get_translation)

st.write("French Predictions:", st.session_state.count)
