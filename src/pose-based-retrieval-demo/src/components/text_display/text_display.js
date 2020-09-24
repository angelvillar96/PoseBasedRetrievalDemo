import React from "react"

class Text extends React.Component{
  constructor(props){
    super(props)
    this.state = {
      font_size: "1.6em",
      font_style: "normal",
      margin: "auto"
    }
    this.update_style = this.update_style.bind(this)
  }

  // processing props for updating the style
  update_style(){
    if(this.props.font_size !== undefined && this.props.font_size !== this.state.font_size){
      this.setState({font_size: this.props.font_size})
    }
    if(this.props.font_style !== undefined && this.props.font_style !== this.state.font_style){
      this.setState({font_style: this.props.font_style})
    }
  }


  render(){
    this.update_style()
    let style = {
      fontSize: this.state.font_size,
      fontStyle: this.state.font_style,
      margin: this.state.margin
    }
    let align = {
      textAlign: this.props.align
    }
    return(
      <div className="text_displays" style={style}>
        {this.props.text_display.split('%aux%').map( (it, i) =>
          <div style={align} key={'x'+i}>{it}</div> )}
      </div>
    )
  }

}

export default Text

//
