import * as firebase from 'firebase'
import 'firebase/firestore'

// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
    apiKey: "AIzaSyD4kGIWN3U9r91vyW-jQtlQCTCEJWilqmU",
    authDomain: "automl-a84ae.firebaseapp.com",
    databaseURL: "https://automl-a84ae.firebaseio.com",
    projectId: "automl-a84ae",
    storageBucket: "automl-a84ae.appspot.com",
    messagingSenderId: "66411754125",
    appId: "1:66411754125:web:f258253cd50192d4a8e8c4",
    measurementId: "G-JNDCK3GDX8"
  };

const firebaseApp = firebase.initializeApp(firebaseConfig);
let db = firebase.firestore();
const settings = {};
db.settings(settings);

// firebase.firestore().enablePersistence()
//   .then(function() {
//     // Initialize Cloud Firestore through firebase
//     db = firebase.firestore();
//   });

const auth = firebase.auth();
const storage = firebase.storage();
const functions = firebaseApp.functions('us-central1');
export {
    auth,
    db,
    functions,
    storage
};
