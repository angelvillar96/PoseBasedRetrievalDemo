import React from "react"

class Text extends React.Component{

    render(){
      return(
        <div className="text_displays">
          {this.props.text_display.split('%aux%').map( (it, i) => <div key={'x'+i}>{it}</div> )}
        </div>
      )
    }

}

export default Text

//
