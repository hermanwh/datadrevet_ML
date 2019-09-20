import React, { Component } from "react";
import "./Landingpage.css";
import Plot from "react-plotly.js";

class LandingPage extends Component {
  constructor(props) {
    super(props);
    this.state = {
      data: [{ y: [0.1, 0.5, 0.7] }],
      layout: {
        width: 420,
        height: 340,
        padding: 0,
        title: "Data from test rigg",
        plot_bgcolor: "#fff"
      },
      frames: [],
      config: {}
    };
  }

  componentDidMount() {
    setInterval(() => {
      var k = this.state.data[0].y.slice();
      k.push(Math.random());
      k.slice(0, 1);
      console.log(k);
      this.setState({
        data: [
          {
            y: k
          }
        ]
      });
    }, 2000);
  }

  getData = () => {
    return Math.random();
  };

  render() {
    return (
      <div className="app__wrapper">
        <div className="header">DATADRIVEN PROJECT</div>
        <div className="sub-header">by Herman Horn and Erik Kjernlie</div>
        <div className="plots">
          <div className="plot__wrapper">
            <Plot
              data={this.state.data}
              layout={this.state.layout}
              frames={this.state.frames}
              config={this.state.config}
              onInitialized={figure => this.setState(figure)}
              onUpdate={figure => this.setState(figure)}
            />
          </div>
        </div>
      </div>
    );
  }
}

export default LandingPage;
