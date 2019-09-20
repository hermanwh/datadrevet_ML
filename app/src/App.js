import React, { Component } from "react";
import "./App.css";
import { Route, Link, BrowserRouter, Switch } from "react-router-dom";

import Upload from "./upload/Upload";
import Landingpage from "./landingpage/Landingpage";

class App extends Component {
  render() {
    return (
      <BrowserRouter>
        <div className="Header">
          <Link to="/" style={{ textDecoration: "none", color: "white" }}>
            Datadriven machine learning
          </Link>
          <Link to="/upload" style={{ textDecoration: "none", color: "white" }}>
            New project
          </Link>
        </div>

        <div className="App">
          <div className="Card">
            <Switch>
              <Route exact path="/" component={Landingpage} />
              <Route path="/upload" component={Upload} />
            </Switch>
          </div>
        </div>
      </BrowserRouter>
    );
  }
}

export default App;
