import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import * as serviceWorker from './serviceWorker';
import 'bootstrap/dist/css/bootstrap.min.css';
import {Switch, BrowserRouter, Route} from 'react-router-dom';

import RetrievalPage from "./demos/retrieval_page/retrieval_page.js";
import IndexPage from "./demos/index_page/index_page.js";


ReactDOM.render((
  <BrowserRouter>
      <Switch>
        <Route exact path="/pose-based-retrieval" component={RetrievalPage}/>
        <Route path="/" component={IndexPage}/>
      </Switch>
    </BrowserRouter>),
  document.getElementById('root')
);

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();
