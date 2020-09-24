import React from "react"

import "./style.css"

class Separator extends React.Component{
  constructor(props){
    super(props)
    this.RenderBox = this.RenderBox.bind(this);
  }

  // rendering box with title if not undefined
  RenderBox(title){
    if(title === undefined){
      return;
    }
    else{
      return(
        <div className="box">
         <div className="title">
           <h4>{this.props.title}</h4>
         </div>
       </div>
     );
   }
  }

  render(){
    return(
      <div className="separator">
        <hr/>
        {this.RenderBox(this.props.title)}

      </div>
    )
  }
}

export default Separator

//
