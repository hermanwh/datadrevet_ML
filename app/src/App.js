import React, { Component } from "react";
import "./App.css";
import { Route, Link, BrowserRouter, Switch } from "react-router-dom";

import Upload from "./upload/Upload";
import Download from "./download/Download";
import Landingpage from "./landingpage/Landingpage";
import CreateProject from "./createProject/CreateProject";

class App extends Component {
  render() {
    return (
      <BrowserRouter>
        <div className="Header">
          <Link to="/" style={{ textDecoration: "none", color: "white" }}>
            Datadriven machine learning
          </Link>
          <Link to="/" style={{ textDecoration: "none", color: "white" }}>
            New project
          </Link>
        </div>
        <Switch>
          <Route exact path="/" component={CreateProject} />
        </Switch>
      </BrowserRouter>
    );
  }
}

export default App;
