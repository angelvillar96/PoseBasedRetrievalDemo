import React from "react"

import arrow_right from "./imgs/arrow_right.jpg"
import arrow_left from "./imgs/arrow_left.jpg"
import "./styles/arrow_button.css"

class ArrowButton extends React.Component{

  render(){

    var img;
    if(this.props.orientation == "left"){
      img = arrow_left;
    }else{
      img = arrow_right;
    }

    return(
      <button className="arrow_button">
        <img src={img} className="arrow_img" alt="arrow_button" onClick={this.props.onClick}/>
      </button>
    )
  };
}

export default ArrowButton
